"""Three-robot autonomous billiards controller for Webots.

Each robot runs this same controller. The robot name decides which ball it owns:
cyan_bot -> cyan ball / blue pocket, yellow_bot -> yellow ball / green pocket,
and magenta_bot -> magenta ball / red pocket.
"""

import math

from controller import Supervisor


ROBOT_Z = 0.34
TIME_LIMIT = 595.0

APPROACH_OFFSET = 1.22
CONTACT_OFFSET = 0.78
PUSH_FINISH_OFFSET = 0.28

MAX_SPEED = 3.0
MAX_PUSH_SPEED = 2.5
MAX_TURN = 6.5
PEER_YIELD_DISTANCE = 0.9
PUSH_START_DISTANCE = 0.96
PURPLE_CLEARANCE = 1.22
BALL_CLEARANCE = 0.98
PEER_CLEARANCE = 1.04
HARD_PURPLE_CLEARANCE = 0.94
HARD_BALL_CLEARANCE = 0.78
HARD_PEER_CLEARANCE = 0.86

ARENA_X_MIN = -6.55
ARENA_X_MAX = 2.55
ARENA_Y_MIN = -3.13
ARENA_Y_MAX = 6.15

GREEN_POCKET = (-6.5, 6.0)
BLUE_POCKET = (2.5, 6.0)
RED_POCKET = (-6.5, -3.0)
POCKETS = (GREEN_POCKET, BLUE_POCKET, RED_POCKET)

BALL_DEFS = {
    "yellow": "YELLOW_BALL",
    "cyan": "CYAN_BALL",
    "magenta": "MAGENTA_BALL",
    "purple": "PURPLE_BALL",
}

BOT_DEFS = {
    "cyan_bot": "CYAN_BOT",
    "yellow_bot": "YELLOW_BOT",
    "magenta_bot": "MAGENTA_BOT",
}

ASSIGNMENTS = {
    "cyan_bot": {
        "ball": "cyan",
        "pocket": BLUE_POCKET,
        "pocket_name": "blue",
        "launch": (2.25, -1.12),
        "priority": 1,
        "start_delay": 0.0,
        "side_bias": -1.0,
    },
    "yellow_bot": {
        "ball": "yellow",
        "pocket": GREEN_POCKET,
        "pocket_name": "green",
        "launch": (1.25, -1.12),
        "priority": 2,
        "start_delay": 0.25,
        "side_bias": 1.0,
    },
    "magenta_bot": {
        "ball": "magenta",
        "pocket": RED_POCKET,
        "pocket_name": "red",
        "launch": (0.65, -2.35),
        "priority": 3,
        "start_delay": 0.5,
        "side_bias": -1.0,
    },
}


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
self_node = robot.getSelf()
translation_field = self_node.getField("translation")
rotation_field = self_node.getField("rotation")
self_name = self_node.getField("name").getSFString()
role = ASSIGNMENTS.get(self_name)

balls = {name: robot.getFromDef(def_name) for name, def_name in BALL_DEFS.items()}
bots = {name: robot.getFromDef(def_name) for name, def_name in BOT_DEFS.items()}

last_debug_time = -99.0
last_state = None


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_point(point, margin=0.0):
    return (
        clamp(point[0], ARENA_X_MIN + margin, ARENA_X_MAX - margin),
        clamp(point[1], ARENA_Y_MIN + margin, ARENA_Y_MAX - margin),
    )


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit_from_to(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (1.0, 0.0)
    return (dx / length, dy / length)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def scale(v, amount):
    return (v[0] * amount, v[1] * amount)


def perp(v):
    return (-v[1], v[0])


def xy_of(node):
    position = node.getPosition()
    return (position[0], position[1])


def robot_xy():
    return xy_of(self_node)


def ball_xy(name):
    return xy_of(balls[name])


def robot_yaw():
    orientation = self_node.getOrientation()
    return math.atan2(orientation[3], orientation[0])


def point_segment_distance(point, start, end):
    sx, sy = start
    ex, ey = end
    px, py = point
    vx = ex - sx
    vy = ey - sy
    length_sq = vx * vx + vy * vy
    if length_sq < 1e-9:
        return distance(point, start), 0.0
    t = ((px - sx) * vx + (py - sy) * vy) / length_sq
    t = clamp(t, 0.0, 1.0)
    closest = (sx + t * vx, sy + t * vy)
    return distance(point, closest), t


def point_path_clearance(point, start, end):
    sx, sy = start
    ex, ey = end
    px, py = point
    vx = ex - sx
    vy = ey - sy
    length_sq = vx * vx + vy * vy
    if length_sq < 1e-9:
        return distance(point, start), 0.0
    raw_t = ((px - sx) * vx + (py - sy) * vy) / length_sq
    t = clamp(raw_t, 0.0, 1.0)
    closest = (sx + t * vx, sy + t * vy)
    return distance(point, closest), raw_t


def pocketed_at(ball_name, pocket):
    position = balls[ball_name].getPosition()
    xy = (position[0], position[1])
    in_square = abs(xy[0] - pocket[0]) <= 0.56 and abs(xy[1] - pocket[1]) <= 0.56
    return in_square or distance(xy, pocket) <= 0.62


def in_any_pocket(ball_name):
    return balls[ball_name].getPosition()[2] < 0.35 or any(pocketed_at(ball_name, pocket) for pocket in POCKETS)


def correct_ball_pocketed():
    position = balls[role["ball"]].getPosition()
    if position[2] > 0.35:
        return False
    return pocketed_at(role["ball"], role["pocket"])


def wrong_pocketed():
    owned = role["ball"]
    position = balls[owned].getPosition()
    if position[2] > 0.35:
        return False
    for pocket in POCKETS:
        if pocket == role["pocket"]:
            continue
        if pocketed_at(owned, pocket):
            return True
    return False


def peer_positions():
    peers = []
    for name, node in bots.items():
        if name != self_name and node is not None:
            peers.append((name, xy_of(node)))
    return peers


def closest_peer():
    mine = robot_xy()
    best = (None, 999.0)
    for name, position in peer_positions():
        d = distance(mine, position)
        if d < best[1]:
            best = (name, d)
    return best


def debug(state, detail="", interval=2.0, force=False):
    global last_debug_time, last_state

    now = robot.getTime()
    if not force and state == last_state and now - last_debug_time < interval:
        return

    owned = role["ball"] if role else "none"
    mine = robot_xy()
    ball = ball_xy(owned) if owned in balls and balls[owned] is not None else (0.0, 0.0)
    purple = ball_xy("purple") if balls.get("purple") is not None else (99.0, 99.0)
    peer_name, peer_d = closest_peer()
    print(
        f"[{self_name}] t={now:5.1f} state={state:<14} "
        f"pos=({mine[0]:.2f},{mine[1]:.2f}) "
        f"{owned}=({ball[0]:.2f},{ball[1]:.2f}) "
        f"purple_d={distance(mine, purple):.2f} "
        f"peer={peer_name}:{peer_d:.2f} {detail}",
        flush=True,
    )
    last_debug_time = now
    last_state = state


def maintain_upright():
    position = self_node.getPosition()
    orientation = self_node.getOrientation()
    if position[2] < 0.22 or position[2] > 0.58 or orientation[8] < 0.75:
        yaw = robot_yaw()
        debug("RESET_UPRIGHT", force=True)
        translation_field.setSFVec3f([position[0], position[1], ROBOT_Z])
        rotation_field.setSFRotation([0.0, 0.0, 1.0, yaw])
        self_node.resetPhysics()


def step_once():
    if robot.getTime() >= TIME_LIMIT:
        return False
    ok = robot.step(timestep) != -1
    maintain_upright()
    return ok


def set_motion(forward_speed, yaw_rate):
    yaw = robot_yaw()
    forward_speed = clamp(forward_speed, -MAX_SPEED, MAX_SPEED)
    yaw_rate = clamp(yaw_rate, -MAX_TURN, MAX_TURN)
    vx = forward_speed * math.cos(yaw)
    vy = forward_speed * math.sin(yaw)
    self_node.setVelocity([vx, vy, 0.0, 0.0, 0.0, yaw_rate])


def set_vector_motion(vector, speed, yaw_rate=0.0):
    length = math.hypot(vector[0], vector[1])
    if length < 1e-6:
        set_motion(0.0, yaw_rate)
        return

    speed = clamp(speed, -MAX_SPEED, MAX_SPEED)
    yaw_rate = clamp(yaw_rate, -MAX_TURN, MAX_TURN)
    vx = vector[0] / length * speed
    vy = vector[1] / length * speed
    self_node.setVelocity([vx, vy, 0.0, 0.0, 0.0, yaw_rate])


def stop_robot(steps=3):
    for _ in range(steps):
        self_node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if not step_once():
            break


def timed_motion(forward_speed, yaw_rate, duration):
    deadline = robot.getTime() + duration
    while robot.getTime() < deadline and step_once():
        set_motion(forward_speed, yaw_rate)
    stop_robot(2)


def blend_angles(a, b, weight):
    weight = clamp(weight, 0.0, 1.0)
    x = (1.0 - weight) * math.cos(a) + weight * math.cos(b)
    y = (1.0 - weight) * math.sin(a) + weight * math.sin(b)
    return math.atan2(y, x)


def wall_push(vector):
    x, y = robot_xy()
    ax, ay = vector
    margin = 0.55
    if x < ARENA_X_MIN + margin:
        ax += 1.7 * (ARENA_X_MIN + margin - x)
    if x > ARENA_X_MAX - margin:
        ax -= 1.7 * (x - (ARENA_X_MAX - margin))
    if y < ARENA_Y_MIN + margin:
        ay += 1.7 * (ARENA_Y_MIN + margin - y)
    if y > ARENA_Y_MAX - margin:
        ay -= 1.7 * (y - (ARENA_Y_MAX - margin))
    return (ax, ay)


def add_repulsion(vector, obstacle, radius, strength):
    mine = robot_xy()
    away = (mine[0] - obstacle[0], mine[1] - obstacle[1])
    d = math.hypot(away[0], away[1])
    if d < 1e-5 or d >= radius:
        return vector
    amount = ((radius - d) / radius) ** 2 * strength
    return (vector[0] + away[0] / d * amount, vector[1] + away[1] / d * amount)


def line_avoidance(vector, goal, obstacle, radius, strength):
    mine = robot_xy()
    line_d, along = point_segment_distance(obstacle, mine, goal)
    if not (0.03 < along < 0.98 and line_d < radius):
        return vector

    path = (goal[0] - mine[0], goal[1] - mine[1])
    normal = perp(path)
    normal_len = math.hypot(normal[0], normal[1])
    if normal_len < 1e-6:
        return vector

    side = 1.0
    cross = path[0] * (obstacle[1] - mine[1]) - path[1] * (obstacle[0] - mine[0])
    if abs(cross) > 1e-6:
        side = -1.0 if cross > 0.0 else 1.0
    else:
        side = role["side_bias"]

    amount = ((radius - line_d) / radius) * strength
    return (
        vector[0] + side * normal[0] / normal_len * amount,
        vector[1] + side * normal[1] / normal_len * amount,
    )


def avoidance_vector(base_vector, goal, skip_ball=None):
    vector = wall_push(base_vector)

    purple = ball_xy("purple")
    if not in_any_pocket("purple"):
        vector = add_repulsion(vector, purple, 2.75, 9.5)
        vector = line_avoidance(vector, goal, purple, 1.7, 8.0)

    for ball_name in ("yellow", "cyan", "magenta"):
        if ball_name == skip_ball or in_any_pocket(ball_name):
            continue
        position = ball_xy(ball_name)
        vector = add_repulsion(vector, position, 1.35, 2.8)
        vector = line_avoidance(vector, goal, position, 1.1, 2.2)

    for peer_name, position in peer_positions():
        peer_priority = ASSIGNMENTS[peer_name]["priority"]
        strength = 4.2 if role["priority"] > peer_priority else 2.4
        vector = add_repulsion(vector, position, 1.45, strength)
        vector = line_avoidance(vector, goal, position, 1.05, 2.4)

    return vector


def should_yield():
    mine = robot_xy()
    for peer_name, position in peer_positions():
        if distance(mine, position) < PEER_YIELD_DISTANCE and role["priority"] > ASSIGNMENTS[peer_name]["priority"]:
            return peer_name
    return None


def purple_too_close():
    if in_any_pocket("purple"):
        return False
    return distance(robot_xy(), ball_xy("purple")) < 1.28


def flee_purple():
    purple = ball_xy("purple")
    mine = robot_xy()
    away = unit_from_to(purple, mine)
    flee_goal = clamp_point(add(mine, scale(away, 1.8)), margin=0.25)
    debug("FLEE_PURPLE", f"goal=({flee_goal[0]:.2f},{flee_goal[1]:.2f})", force=True)
    deadline = robot.getTime() + 1.25
    while robot.getTime() < deadline and step_once():
        drive_step(flee_goal, avoid=True, skip_ball=role["ball"], max_speed=1.05)
    stop_robot(2)


def yield_to_peer(peer_name):
    debug("YIELD", f"to={peer_name}", force=True)
    timed_motion(-0.22, role["side_bias"] * 1.35, 0.55)


def rescue_from_stall(goal):
    mine = robot_xy()
    away_from_goal = unit_from_to(goal, mine)
    rescue_goal = clamp_point(add(mine, scale(away_from_goal, 0.9)), margin=0.22)
    debug("UNSTICK", f"rescue=({rescue_goal[0]:.2f},{rescue_goal[1]:.2f})", force=True)
    timed_motion(-0.45, role["side_bias"] * 1.8, 0.7)
    deadline = robot.getTime() + 0.85
    while robot.getTime() < deadline and step_once():
        drive_step(rescue_goal, avoid=True, skip_ball=role["ball"], max_speed=0.45)
    stop_robot(2)


def drive_step(goal, desired_heading=None, avoid=True, skip_ball=None, max_speed=0.72, heading_bias=0.0):
    position = robot_xy()
    dx = goal[0] - position[0]
    dy = goal[1] - position[1]
    dist = math.hypot(dx, dy)

    if dist > 1e-5:
        nav_vector = (dx, dy)
        if avoid:
            nav_vector = avoidance_vector(nav_vector, goal, skip_ball)
        nav_heading = math.atan2(nav_vector[1], nav_vector[0])
    else:
        nav_heading = robot_yaw()

    if desired_heading is not None:
        align_window = 1.0 + heading_bias
        near_weight = clamp((align_window - dist) / align_window, 0.0, 1.0)
        nav_heading = blend_angles(nav_heading, desired_heading, near_weight)

    heading_error = wrap_angle(nav_heading - robot_yaw())
    turn = clamp(4.8 * heading_error, -MAX_TURN, MAX_TURN)

    speed = min(max_speed, 0.38 + 1.75 * dist)
    if abs(heading_error) > 1.55:
        speed = 0.0
    else:
        speed *= max(0.32, math.cos(heading_error))
    if dist < 0.24:
        speed *= max(0.5, dist / 0.24)

    if not in_any_pocket("purple"):
        purple_d = distance(position, ball_xy("purple"))
        if purple_d < 1.15:
            speed *= 0.55

    set_motion(speed, turn)
    return dist, abs(heading_error)


def drive_to(goal, state, tolerance=0.24, timeout=18.0, desired_heading=None, avoid=True, skip_ball=None, max_speed=0.72, heading_bias=0.0):
    goal = clamp_point(goal, margin=0.05)
    deadline = robot.getTime() + timeout
    best_dist = float("inf")
    best_time = robot.getTime()

    debug(state, f"goal=({goal[0]:.2f},{goal[1]:.2f})", force=True)
    while robot.getTime() < deadline and step_once():
        if purple_too_close():
            flee_purple()
            best_time = robot.getTime()

        peer_name = should_yield()
        if peer_name:
            yield_to_peer(peer_name)
            best_time = robot.getTime()

        dist, heading_error = drive_step(
            goal,
            desired_heading=desired_heading,
            avoid=avoid,
            skip_ball=skip_ball,
            max_speed=max_speed,
            heading_bias=heading_bias,
        )
        debug(state, f"dist={dist:.2f} heading_error={heading_error:.2f}")

        if dist < best_dist - 0.035:
            best_dist = dist
            best_time = robot.getTime()

        aligned = desired_heading is None or heading_error < 0.24
        if dist <= tolerance and aligned:
            stop_robot(3)
            return True

        if robot.getTime() - best_time > 1.8 and dist > tolerance + 0.12:
            rescue_from_stall(goal)
            best_time = robot.getTime()

    stop_robot(3)
    debug("TIMEOUT", f"state={state} goal=({goal[0]:.2f},{goal[1]:.2f})", force=True)
    return False


def turn_to(heading, timeout=1.6, tolerance=0.16):
    deadline = robot.getTime() + timeout
    debug("TURN", f"heading={heading:.2f}", force=True)
    while robot.getTime() < deadline and step_once():
        error = wrap_angle(heading - robot_yaw())
        if abs(error) < tolerance:
            stop_robot(3)
            return True
        set_motion(0.0, clamp(5.4 * error, -MAX_TURN, MAX_TURN))
    stop_robot(3)
    return False


def shot_lane_blockers(
    start,
    end,
    purple_clearance=PURPLE_CLEARANCE,
    ball_clearance=BALL_CLEARANCE,
    peer_clearance=PEER_CLEARANCE,
):
    blockers = []

    for ball_name in ("purple", "yellow", "cyan", "magenta"):
        if ball_name == role["ball"] or in_any_pocket(ball_name):
            continue
        position = ball_xy(ball_name)
        clearance, along = point_path_clearance(position, start, end)
        limit = purple_clearance if ball_name == "purple" else ball_clearance
        start_limit = limit * (0.78 if ball_name != "purple" else 0.92)
        blocks_start = -0.08 < along < 0.08 and clearance < start_limit
        blocks_lane = 0.08 <= along < 0.98 and clearance < limit
        if blocks_start or blocks_lane:
            blockers.append((ball_name, position, clearance, along, limit))

    for peer_name, position in peer_positions():
        clearance, along = point_path_clearance(position, start, end)
        limit = peer_clearance
        if 0.0 < along < 0.98 and clearance < limit:
            blockers.append((peer_name, position, clearance, along, limit))

    return blockers


def shot_lane_blocker(
    start,
    end,
    purple_clearance=PURPLE_CLEARANCE,
    ball_clearance=BALL_CLEARANCE,
    peer_clearance=PEER_CLEARANCE,
):
    best = None
    for candidate in shot_lane_blockers(start, end, purple_clearance, ball_clearance, peer_clearance):
        if best is None or candidate[3] < best[3]:
            best = candidate

    return best


def hard_shot_lane_blocker(start, end):
    return shot_lane_blocker(start, end, HARD_PURPLE_CLEARANCE, HARD_BALL_CLEARANCE, HARD_PEER_CLEARANCE)


def blocker_cost(blockers):
    cost = 0.0
    for name, _, clearance, along, limit in blockers:
        shortage = max(0.0, limit - clearance)
        weight = 42.0 if name == "purple" else 24.0 if name in BALL_DEFS else 16.0
        near_weight = 1.25 - 0.45 * clamp(along, 0.0, 1.0)
        cost += weight * near_weight * (0.2 + shortage + shortage * shortage)
    return cost


def aim_point(ball, pocket):
    direction = unit_from_to(ball, pocket)
    normal = perp(direction)

    candidates = [(pocket, 0.0)]
    for offset in (1.15, 1.65, 2.2, 2.8):
        candidates.append((clamp_point(add(pocket, scale(normal, role["side_bias"] * offset)), margin=0.25), offset))
        candidates.append((clamp_point(add(pocket, scale(normal, -role["side_bias"] * offset)), margin=0.25), offset))

    best_aim = pocket
    best_cost = float("inf")
    direct_blocked = shot_lane_blocker(ball, pocket) is not None

    for candidate, offset in candidates:
        blockers = shot_lane_blockers(ball, candidate)
        cost = blocker_cost(blockers) + distance(candidate, pocket) * 0.55 + offset * 0.35
        if blockers:
            cost += 3.0
        if cost < best_cost:
            best_cost = cost
            best_aim = candidate

    return best_aim, direct_blocked or distance(best_aim, pocket) > 0.2


def stage_candidates(ball, direction):
    base = add(ball, scale(direction, -APPROACH_OFFSET))
    normal = perp(direction)
    candidates = []
    for lateral in (0.0, 0.28, -0.28, 0.56, -0.56, 0.84, -0.84, 1.12, -1.12):
        point = clamp_point(add(base, scale(normal, lateral)), margin=0.22)
        candidates.append((point, abs(lateral)))
    return candidates


def obstacle_cost(point):
    cost = 0.0
    purple = ball_xy("purple")
    if not in_any_pocket("purple"):
        d = distance(point, purple)
        if d < 2.05:
            cost += (2.05 - d) * 22.0

    for peer_name, position in peer_positions():
        d = distance(point, position)
        if d < 1.35:
            cost += (1.35 - d) * 10.0

    for ball_name in ("yellow", "cyan", "magenta"):
        if ball_name == role["ball"] or in_any_pocket(ball_name):
            continue
        d = distance(point, ball_xy(ball_name))
        if d < 1.25:
            cost += (1.25 - d) * 8.0

    return cost


def shot_geometry():
    owned_ball = ball_xy(role["ball"])
    pocket = role["pocket"]
    aim, detour = aim_point(owned_ball, pocket)
    direction = unit_from_to(owned_ball, aim)
    heading = math.atan2(direction[1], direction[0])

    best_stage = None
    best_cost = float("inf")
    for candidate, lateral in stage_candidates(owned_ball, direction):
        cost = distance(robot_xy(), candidate) + obstacle_cost(candidate) + lateral * 2.4
        if cost < best_cost:
            best_cost = cost
            best_stage = candidate

    contact = clamp_point(add(owned_ball, scale(direction, -CONTACT_OFFSET)), margin=0.18)
    push_goal = clamp_point(add(aim, scale(direction, -PUSH_FINISH_OFFSET)), margin=0.15)
    return owned_ball, pocket, aim, detour, direction, heading, best_stage, contact, push_goal


def clear_blocked_lane(blocker_name, blocker_position, direction):
    mine = robot_xy()
    normal = perp(direction)
    away = (mine[0] - blocker_position[0], mine[1] - blocker_position[1])
    side = 1.0 if away[0] * normal[0] + away[1] * normal[1] >= 0.0 else -1.0
    if abs(away[0] * normal[0] + away[1] * normal[1]) < 1e-5:
        side = role["side_bias"]

    clear_goal = clamp_point(
        add(add(mine, scale(normal, side * 0.85)), scale(direction, -0.3)),
        margin=0.25,
    )
    debug("CLEAR_PATH", f"{blocker_name} goal=({clear_goal[0]:.2f},{clear_goal[1]:.2f})", force=True)
    deadline = robot.getTime() + 0.65
    while robot.getTime() < deadline and step_once():
        drive_step(clear_goal, avoid=True, skip_ball=role["ball"], max_speed=1.0)
    stop_robot(2)


def assigned_ball_in_bad_state():
    if wrong_pocketed():
        debug("WRONG_POCKET", f"{role['ball']} is not in {role['pocket_name']}", force=True)
        return True
    return False


def push_assigned_ball():
    attempts = 0
    while robot.getTime() < TIME_LIMIT and not correct_ball_pocketed():
        if assigned_ball_in_bad_state():
            return False

        attempts += 1
        ball, pocket, aim, detour, direction, heading, stage, contact, push_goal = shot_geometry()
        detail = (
            f"ball=({ball[0]:.2f},{ball[1]:.2f}) "
            f"stage=({stage[0]:.2f},{stage[1]:.2f}) "
            f"aim=({aim[0]:.2f},{aim[1]:.2f}) detour={detour}"
        )
        debug("PLAN_SHOT", detail, force=True)

        drive_to(
            stage,
            "APPROACH",
            tolerance=0.34,
            timeout=10.0,
            avoid=True,
            skip_ball=role["ball"],
            max_speed=2.5,
        )
        turn_to(heading)

        push_deadline = robot.getTime() + 7.5
        last_ball = ball_xy(role["ball"])
        last_ball_move = robot.getTime()

        while robot.getTime() < push_deadline and step_once():
            if correct_ball_pocketed():
                debug("SCORED", f"{role['ball']} -> {role['pocket_name']}", force=True)
                stop_robot(8)
                return True

            if assigned_ball_in_bad_state():
                stop_robot(8)
                return False

            if purple_too_close():
                flee_purple()
                push_deadline += 1.0

            peer_name = should_yield()
            if peer_name and distance(robot_xy(), contact) > 0.55:
                yield_to_peer(peer_name)
                push_deadline += 0.7

            ball, pocket, aim, detour, direction, heading, stage, contact, push_goal = shot_geometry()
            robot_to_contact = distance(robot_xy(), contact)
            angle_error = abs(wrap_angle(heading - robot_yaw()))
            blocker = hard_shot_lane_blocker(ball, aim)

            if robot_to_contact > PUSH_START_DISTANCE:
                to_contact = unit_from_to(robot_xy(), contact)
                yaw_error = wrap_angle(heading - robot_yaw())
                set_vector_motion(to_contact, 2.0, 5.2 * yaw_error)
                debug("REACQUIRE", f"contact={robot_to_contact:.2f} angle={angle_error:.2f}")
            elif angle_error > 0.42:
                error = wrap_angle(heading - robot_yaw())
                set_motion(0.0, clamp(5.8 * error, -MAX_TURN, MAX_TURN))
                debug("AIM", f"contact={robot_to_contact:.2f} angle={angle_error:.2f}")
            elif blocker is not None:
                name, position, clearance, along, _ = blocker
                debug("LANE_BLOCKED", f"{name} clearance={clearance:.2f} along={along:.2f}", force=True)
                if name in BOT_DEFS:
                    yield_to_peer(name)
                else:
                    clear_blocked_lane(name, position, direction)
                break
            else:
                yaw_error = wrap_angle(heading - robot_yaw())
                push_speed = MAX_PUSH_SPEED if distance(ball, pocket) > 1.0 else 1.6
                set_vector_motion(direction, push_speed, 4.6 * yaw_error)
                debug("PUSH", f"pocket_d={distance(ball, pocket):.2f} detour={detour}")

            current_ball = ball_xy(role["ball"])
            engaged = robot_to_contact < PUSH_START_DISTANCE and angle_error < 0.5
            if not engaged:
                last_ball = current_ball
                last_ball_move = robot.getTime()
            elif distance(current_ball, last_ball) > 0.08:
                last_ball = current_ball
                last_ball_move = robot.getTime()
            elif robot.getTime() - last_ball_move > 2.0:
                debug("BALL_STALLED", "backing up for another approach", force=True)
                break

        stop_robot(3)
        timed_motion(-0.45, role["side_bias"] * 2.2, 0.35)

        if attempts >= 10:
            debug("REPLAN_LONG", "many attempts, continuing carefully", force=True)
            attempts = 0

    return correct_ball_pocketed()


def ready():
    if role is None:
        print(f"[{self_name}] no assignment for this robot name", flush=True)
        return False
    missing = [name for name, node in balls.items() if node is None]
    missing += [name for name, node in bots.items() if node is None]
    if missing:
        print(f"[{self_name}] missing DEF nodes: {', '.join(missing)}", flush=True)
        return False
    return True


if ready():
    debug("BOOT", f"assigned={role['ball']}->{role['pocket_name']}", force=True)
    while robot.getTime() < role["start_delay"] and step_once():
        stop_robot(1)

    drive_to(
        role["launch"],
        "LAUNCH",
        tolerance=0.38,
        timeout=2.8,
        avoid=True,
        skip_ball=role["ball"],
        max_speed=2.2,
    )
    push_assigned_ball()
    debug("DONE", f"scored={correct_ball_pocketed()}", force=True)
    stop_robot(30)
else:
    stop_robot(20)