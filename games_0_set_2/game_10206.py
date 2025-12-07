import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:05:37.368149
# Source Brief: brief_00206.md
# Brief Index: 206
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    GameEnv: Robot Path Optimizer
    An environment where an agent configures the paths and speeds of oscillating robots
    to maximize the number that reach their destinations within a time limit.

    The agent operates in two modes:
    1. EDIT MODE: The agent uses actions to select robots, adjust their oscillation
       speed, and move the two endpoints of their path.
    2. SIMULATION MODE: Triggered by a 'no-op' action, this mode runs a 30-second
       simulation of the robot movements based on the current configuration. The
       agent observes the simulation unfold and receives rewards based on performance.
       The episode terminates at the end of the simulation.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Configure robot paths and speeds to guide as many as possible to their "
        "destinations without collisions in a timed simulation."
    )
    user_guide = (
        "Use ↑↓ to select a robot and ←→ to adjust speed. Hold space + arrows to move the path "
        "start point, or shift + arrows for the end point. Submit a no-op (no keys) to run the simulation."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_MARGIN = 40
    GRID_WIDTH = SCREEN_WIDTH - 2 * GRID_MARGIN
    GRID_HEIGHT = SCREEN_HEIGHT - 2 * GRID_MARGIN
    NUM_ROBOTS = 10
    ROBOT_SIZE = 8
    PATH_POINT_SIZE = 5
    DESTINATION_SIZE = 10
    SUCCESS_RADIUS = 12
    MAX_SIM_TIME = 30.0  # seconds
    SIM_TIME_STEP = 1.0 / 30.0  # 30 FPS simulation
    MAX_EPISODE_STEPS = 500 # Max commands before termination
    MIN_SPEED = 0.5
    MAX_SPEED = 5.0
    SPEED_INCREMENT = 0.1
    PATH_MOVE_SPEED = 5

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    COLOR_UI_BG = (10, 20, 30, 180)
    COLOR_TEXT = (220, 220, 240)
    COLOR_PATH = (80, 100, 120)
    COLOR_PATH_POINT = (180, 180, 180)
    COLOR_DESTINATION = (255, 220, 0)
    COLOR_SELECTION = (0, 200, 255)
    ROBOT_COLORS = {
        "active": (0, 255, 128),
        "collided": (255, 80, 80),
        "success": (80, 150, 255),
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        # --- Internal State ---
        self.mode = 'edit'
        self.steps = 0
        self.score = 0
        self.sim_time = 0.0
        self.selected_robot_idx = 0
        self.robots = []
        self.last_distances = {}

        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.mode = 'edit'
        self.steps = 0
        self.score = 0
        self.sim_time = 0.0
        self.selected_robot_idx = 0
        self.robots = []
        self.last_distances = {}

        for i in range(self.NUM_ROBOTS):
            p1 = pygame.math.Vector2(
                self.np_random.uniform(self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_WIDTH),
                self.np_random.uniform(self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_HEIGHT)
            )
            p2 = pygame.math.Vector2(
                self.np_random.uniform(self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_WIDTH),
                self.np_random.uniform(self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_HEIGHT)
            )
            dest = pygame.math.Vector2(
                self.np_random.uniform(self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_WIDTH),
                self.np_random.uniform(self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_HEIGHT)
            )
            self.robots.append({
                "id": i,
                "p1": p1,
                "p2": p2,
                "destination": dest,
                "speed": self.np_random.uniform(1.0, 3.0),
                # Simulation state
                "pos": pygame.math.Vector2(p1),
                "t": 0.0,
                "direction": 1,
                "status": "active"
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = False

        if self.mode == 'edit':
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            if movement == 0 and not space_held and not shift_held:
                # ACTION: Run Simulation
                self.mode = 'simulate'
                self._prepare_for_simulation()
                # sound: "simulation_start.wav"
            else:
                self._handle_edit_action(movement, space_held, shift_held)

        elif self.mode == 'simulate':
            reward, sim_is_over = self._run_simulation_step()
            if sim_is_over:
                terminated = True
                successful_robots = sum(1 for r in self.robots if r['status'] == 'success')
                reward += 10 * successful_robots
                self.score = successful_robots
                # sound: "simulation_complete.wav"

        if self.steps >= self.MAX_EPISODE_STEPS:
            truncated = True
            if self.mode == 'simulate':
                successful_robots = sum(1 for r in self.robots if r['status'] == 'success')
                reward += 10 * successful_robots
                self.score = successful_robots

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _prepare_for_simulation(self):
        self.sim_time = 0.0
        self.last_distances.clear()
        for robot in self.robots:
            robot["pos"] = pygame.math.Vector2(robot["p1"])
            robot["status"] = "active"
            robot["t"] = 0.0
            robot["direction"] = 1
            self.last_distances[robot["id"]] = robot["pos"].distance_to(robot["destination"])

    def _handle_edit_action(self, movement, space_held, shift_held):
        selected_robot = self.robots[self.selected_robot_idx]

        # --- Path Point Adjustment ---
        if space_held or shift_held:
            target_point = "p1" if space_held else "p2"
            dx, dy = 0, 0
            if movement == 1: dy = -self.PATH_MOVE_SPEED  # Up
            elif movement == 2: dy = self.PATH_MOVE_SPEED   # Down
            elif movement == 3: dx = -self.PATH_MOVE_SPEED  # Left
            elif movement == 4: dx = self.PATH_MOVE_SPEED   # Right

            if dx != 0 or dy != 0:
                # sound: "path_point_move.wav"
                selected_robot[target_point].x += dx
                selected_robot[target_point].y += dy
                # Clamp to grid
                selected_robot[target_point].x = np.clip(selected_robot[target_point].x, self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_WIDTH)
                selected_robot[target_point].y = np.clip(selected_robot[target_point].y, self.GRID_MARGIN, self.GRID_MARGIN + self.GRID_HEIGHT)
            return

        # --- Robot Selection & Speed Adjustment ---
        if movement == 1:  # Up to select previous
            self.selected_robot_idx = (self.selected_robot_idx - 1) % self.NUM_ROBOTS
            # sound: "select_robot.wav"
        elif movement == 2:  # Down to select next
            self.selected_robot_idx = (self.selected_robot_idx + 1) % self.NUM_ROBOTS
            # sound: "select_robot.wav"
        elif movement == 3:  # Left to decrease speed
            selected_robot["speed"] = max(self.MIN_SPEED, selected_robot["speed"] - self.SPEED_INCREMENT)
            # sound: "speed_down.wav"
        elif movement == 4:  # Right to increase speed
            selected_robot["speed"] = min(self.MAX_SPEED, selected_robot["speed"] + self.SPEED_INCREMENT)
            # sound: "speed_up.wav"

    def _run_simulation_step(self):
        self.sim_time += self.SIM_TIME_STEP
        incremental_reward = 0.0

        # 1. Update positions
        for robot in self.robots:
            if robot["status"] != "active":
                continue

            path_vector = robot["p2"] - robot["p1"]
            path_length = path_vector.length()

            if path_length > 1e-6:
                delta_t = (robot["speed"] * self.SIM_TIME_STEP) / path_length
                robot["t"] += robot["direction"] * delta_t
                if robot["t"] >= 1.0:
                    robot["t"] = 1.0
                    robot["direction"] = -1
                elif robot["t"] <= 0.0:
                    robot["t"] = 0.0
                    robot["direction"] = 1
                robot["pos"] = robot["p1"] + robot["t"] * path_vector

            # Check for success
            dist_to_dest = robot["pos"].distance_to(robot["destination"])
            if dist_to_dest < self.SUCCESS_RADIUS:
                robot["status"] = "success"
                incremental_reward += 1.0
                # sound: "robot_success.wav"

            # Reward for getting closer
            if dist_to_dest < self.last_distances[robot["id"]]:
                incremental_reward += 0.1
            self.last_distances[robot["id"]] = dist_to_dest

        # 2. Check for collisions
        active_robots = [r for r in self.robots if r["status"] == "active"]
        for i in range(len(active_robots)):
            for j in range(i + 1, len(active_robots)):
                r1 = active_robots[i]
                r2 = active_robots[j]
                if r1["pos"].distance_to(r2["pos"]) < self.ROBOT_SIZE * 2:
                    if r1["status"] == "active":
                        r1["status"] = "collided"
                        incremental_reward -= 0.5
                    if r2["status"] == "active":
                        r2["status"] = "collided"
                        incremental_reward -= 0.5
                    # sound: "robot_collision.wav"

        sim_is_over = self.sim_time >= self.MAX_SIM_TIME
        return incremental_reward, sim_is_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _render_game(self):
        # Draw grid
        for i in range(0, self.GRID_WIDTH + 1, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN + i, self.GRID_MARGIN), (self.GRID_MARGIN + i, self.GRID_MARGIN + self.GRID_HEIGHT))
        for i in range(0, self.GRID_HEIGHT + 1, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN, self.GRID_MARGIN + i), (self.GRID_MARGIN + self.GRID_WIDTH, self.GRID_MARGIN + i))

        # Draw elements for each robot
        for i, robot in enumerate(self.robots):
            is_selected = (i == self.selected_robot_idx) and (self.mode == 'edit')

            # Draw destination
            dest_pos = (int(robot["destination"].x), int(robot["destination"].y))
            pygame.gfxdraw.aacircle(self.screen, dest_pos[0], dest_pos[1], self.DESTINATION_SIZE, self.COLOR_DESTINATION)
            pygame.gfxdraw.filled_circle(self.screen, dest_pos[0], dest_pos[1], self.DESTINATION_SIZE, self.COLOR_DESTINATION)

            if self.mode == 'edit':
                # Draw path line
                p1_pos = (int(robot["p1"].x), int(robot["p1"].y))
                p2_pos = (int(robot["p2"].x), int(robot["p2"].y))
                pygame.draw.aaline(self.screen, self.COLOR_PATH, p1_pos, p2_pos)

                # Draw path points
                for p_pos in [p1_pos, p2_pos]:
                    if is_selected:
                        pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], self.PATH_POINT_SIZE + 2, self.COLOR_SELECTION)
                    pygame.gfxdraw.filled_circle(self.screen, p_pos[0], p_pos[1], self.PATH_POINT_SIZE, self.COLOR_PATH_POINT)

            # Draw robot
            robot_pos = (int(robot["pos"].x), int(robot["pos"].y))
            robot_color = self.ROBOT_COLORS[robot["status"]]

            if is_selected:
                pygame.gfxdraw.filled_circle(self.screen, robot_pos[0], robot_pos[1], self.ROBOT_SIZE + 4, self.COLOR_SELECTION)

            robot_rect = pygame.Rect(robot_pos[0] - self.ROBOT_SIZE, robot_pos[1] - self.ROBOT_SIZE, self.ROBOT_SIZE * 2, self.ROBOT_SIZE * 2)
            pygame.draw.rect(self.screen, robot_color, robot_rect, border_radius=3)
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in robot_color), robot_rect, width=2, border_radius=3)

    def _render_ui(self):
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, self.GRID_MARGIN + 10), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, self.SCREEN_HEIGHT - self.GRID_MARGIN - 10))

        # Mode Text
        mode_text = f"MODE: {self.mode.upper()}"
        mode_surf = self.font_main.render(mode_text, True, self.COLOR_TEXT)
        self.screen.blit(mode_surf, (15, self.SCREEN_HEIGHT - self.GRID_MARGIN))

        # Sim Time Text
        sim_time_text = f"SIM TIME: {self.sim_time:.1f}s / {self.MAX_SIM_TIME:.1f}s"
        sim_surf = self.font_main.render(sim_time_text, True, self.COLOR_TEXT)
        self.screen.blit(sim_surf, (180, self.SCREEN_HEIGHT - self.GRID_MARGIN))

        # Score Text
        score_text = f"SUCCESSFUL: {self.score}/{self.NUM_ROBOTS}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (450, self.SCREEN_HEIGHT - self.GRID_MARGIN))

        if self.mode == 'edit':
            selected_robot = self.robots[self.selected_robot_idx]
            info_text = f"EDITING ROBOT #{self.selected_robot_idx+1} | SPEED: {selected_robot['speed']:.1f}"
            info_surf = self.font_small.render(info_text, True, self.COLOR_SELECTION)
            self.screen.blit(info_surf, (15, 10))

            controls_text = "ARROWS: Select/Speed | HOLD SPACE/SHIFT + ARROWS: Move Path Points | NO-OP: Run Sim"
            controls_surf = self.font_small.render(controls_text, True, self.COLOR_TEXT)
            self.screen.blit(controls_surf, (15, self.SCREEN_HEIGHT - 20))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sim_time": self.sim_time,
            "mode": self.mode,
            "selected_robot": self.selected_robot_idx
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block is for manual play and will not be run by the evaluation system.
    # It is provided for your convenience to test the environment.
    # To use, you must have pygame installed.
    
    # Un-comment the following line to run in a window
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # We need to create a display for the manual play, but not for the headless evaluation
    try:
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Robot Path Optimizer")
        is_headless = False
    except pygame.error:
        print("Pygame display could not be initialized. Running in headless mode.")
        is_headless = True


    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    while not terminated:
        action = [0, 0, 0] # Default no-op action
        
        # In a real display, we can check for key presses
        if not is_headless:
            # --- Action Mapping for Human Player ---
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]

            # --- Pygame Event Handling ---
            should_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    should_step = True
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
            
            # In sim mode, we step automatically. In edit mode, we wait for an action.
            if info['mode'] == 'simulate' or should_step:
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    print(f"Episode finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

            # --- Rendering ---
            # The observation is already a rendered frame
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            clock.tick(30) # Limit frame rate for human play
        else: # In headless mode, we can't get key presses, so we just step with no-ops
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Headless Episode finished. Final Score: {info['score']}")


    env.close()