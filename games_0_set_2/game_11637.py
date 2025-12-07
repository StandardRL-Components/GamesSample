import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:20:38.204117
# Source Brief: brief_01637.md
# Brief Index: 1637
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math

class GameEnv(gym.Env):
    """
    A Gymnasium environment where an agent controls two energy-dependent robots.
    The goal is to collect 10 power-ups before running out of energy or time.
    The agent controls the shared movement direction and triggers each robot's
    movement independently, creating a strategic challenge of synchronized
    pathfinding and resource management.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement Direction (0:None, 1:Up, 2:Down, 3:Left, 4:Right)
    - actions[1]: Move Robot 1 (Cyan) (0:No, 1:Yes)
    - actions[2]: Move Robot 2 (Magenta) (0:No, 1:Yes)

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - RGB Array
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Control two energy-dependent robots to collect 10 power-ups before time or energy "
        "runs out, while avoiding hazardous obstacles."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to set movement direction. Press space to move the cyan robot "
        "and shift to move the magenta robot."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1800  # 60 seconds at 30 FPS

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (30, 30, 45)
    COLOR_ROBOT1 = (0, 255, 255)
    COLOR_ROBOT2 = (255, 0, 255)
    COLOR_POWERUP = (255, 255, 0)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_ENERGY_FULL = (50, 255, 50)
    COLOR_ENERGY_EMPTY = (255, 50, 50)

    # Game Parameters
    ROBOT_SIZE = 12
    ROBOT_SPEED = 3.0
    POWERUP_SIZE = 8
    OBSTACLE_SIZE = 16
    INITIAL_OBSTACLES = 10
    INITIAL_ENERGY = 100.0
    MOVE_ENERGY_COST = 0.08
    OBSTACLE_HIT_ENERGY_COST = 15.0
    WIN_CONDITION = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # State Variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.num_obstacles = self.INITIAL_OBSTACLES

        self.robot1 = None
        self.robot2 = None
        self.powerups = []
        self.obstacles = []
        self.particles = []
        self.target_direction = np.array([0.0, 0.0])
        self.last_reward = 0.0

        # This is a placeholder for the first call to reset()
        # self.reset() # Removed to avoid calling reset before seed is properly set by wrapper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.num_obstacles = self.INITIAL_OBSTACLES
        self.target_direction = np.array([0.0, 0.0])
        self.last_reward = 0.0

        self.particles.clear()

        # --- Robot Initialization ---
        # Store previous distances for reward calculation
        self.robot1 = {
            "pos": self._get_random_pos(), "energy": self.INITIAL_ENERGY, "dist_to_powerup": float('inf')
        }
        self.robot2 = {
            "pos": self._get_random_pos(), "energy": self.INITIAL_ENERGY, "dist_to_powerup": float('inf')
        }

        # --- Obstacle and Power-up Generation ---
        self.obstacles.clear()
        for _ in range(self.num_obstacles):
            self._spawn_obstacle()

        self.powerups.clear()
        self._spawn_powerup()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, move_r1, move_r2 = action
        self._update_target_direction(movement)

        reward = 0

        # --- Game Logic Update ---
        # Store distances before moving for reward calculation
        self._update_dist_to_powerups()

        old_dist_r1 = self.robot1["dist_to_powerup"]
        old_dist_r2 = self.robot2["dist_to_powerup"]

        # Move robots
        if move_r1 and self.robot1["energy"] > 0:
            self._move_robot(self.robot1)
        if move_r2 and self.robot2["energy"] > 0:
            self._move_robot(self.robot2)

        # Update distances after moving
        self._update_dist_to_powerups()
        new_dist_r1 = self.robot1["dist_to_powerup"]
        new_dist_r2 = self.robot2["dist_to_powerup"]

        # Continuous reward for getting closer
        if move_r1:
            reward += (old_dist_r1 - new_dist_r1) * 0.01 # Reward for getting closer
        if move_r2:
            reward += (old_dist_r2 - new_dist_r2) * 0.01

        # Check collisions
        reward += self._handle_collisions(self.robot1)
        reward += self._handle_collisions(self.robot2)

        # Update particles
        self._update_particles()

        self.steps += 1

        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.win:
                reward += 100.0  # Goal-oriented win reward
            elif not truncated: # Penalty if lost, but not due to time out
                 reward -= 10.0 # Goal-oriented lose penalty

        self.last_reward = reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _update_target_direction(self, movement_action):
        if movement_action == 1:  # Up
            self.target_direction = np.array([0, -1])
        elif movement_action == 2:  # Down
            self.target_direction = np.array([0, 1])
        elif movement_action == 3:  # Left
            self.target_direction = np.array([-1, 0])
        elif movement_action == 4:  # Right
            self.target_direction = np.array([1, 0])
        else:  # None
            self.target_direction = np.array([0, 0])

    def _move_robot(self, robot):
        if np.linalg.norm(self.target_direction) > 0:
            robot["pos"] += self.target_direction * self.ROBOT_SPEED
            robot["energy"] = max(0, robot["energy"] - self.MOVE_ENERGY_COST)
            # Boundary checks
            robot["pos"][0] = np.clip(robot["pos"][0], 0, self.SCREEN_WIDTH)
            robot["pos"][1] = np.clip(robot["pos"][1], 0, self.SCREEN_HEIGHT)
            # sfx: robot_move_hum

    def _handle_collisions(self, robot):
        reward = 0
        robot_rect = pygame.Rect(robot["pos"][0] - self.ROBOT_SIZE / 2, robot["pos"][1] - self.ROBOT_SIZE / 2, self.ROBOT_SIZE, self.ROBOT_SIZE)

        # Obstacle collisions
        for i, obs_pos in reversed(list(enumerate(self.obstacles))):
            obs_rect = pygame.Rect(obs_pos[0] - self.OBSTACLE_SIZE / 2, obs_pos[1] - self.OBSTACLE_SIZE / 2, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE)
            if robot_rect.colliderect(obs_rect):
                robot["energy"] = max(0, robot["energy"] - self.OBSTACLE_HIT_ENERGY_COST)
                reward -= 5.0  # Event-based penalty for collision
                self._spawn_particles(robot["pos"], self.COLOR_OBSTACLE, 20, 4.0)
                # sfx: obstacle_hit
                # Push robot out of obstacle slightly
                robot["pos"] -= self.target_direction * self.ROBOT_SPEED * 1.5

        # Power-up collisions
        for i, p_pos in reversed(list(enumerate(self.powerups))):
            p_rect = pygame.Rect(p_pos[0] - self.POWERUP_SIZE, p_pos[1] - self.POWERUP_SIZE, self.POWERUP_SIZE * 2, self.POWERUP_SIZE * 2)
            if robot_rect.colliderect(p_rect):
                self.powerups.pop(i)
                self.score += 1
                reward += 10.0  # Event-based reward for collection
                self._spawn_powerup()
                self._spawn_particles(p_pos, self.COLOR_POWERUP, 30, 5.0)
                # sfx: powerup_collect

                # Increase difficulty
                if self.score > 0 and self.score % 2 == 0:
                    self.num_obstacles = int(self.num_obstacles * 1.05)
                    self._spawn_obstacle()

        return reward

    def _check_termination(self):
        if self.score >= self.WIN_CONDITION:
            self.win = True
            return True
        if self.robot1["energy"] <= 0 and self.robot2["energy"] <= 0:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot1_energy": self.robot1["energy"],
            "robot2_energy": self.robot2["energy"],
        }

    def _get_random_pos(self):
        return np.array([
            self.np_random.uniform(low=20, high=self.SCREEN_WIDTH - 20),
            self.np_random.uniform(low=20, high=self.SCREEN_HEIGHT - 20)
        ])

    def _spawn_powerup(self):
        self.powerups.append(self._get_random_pos())

    def _spawn_obstacle(self):
        self.obstacles.append(self._get_random_pos())

    def _update_dist_to_powerups(self):
        if not self.powerups:
            self.robot1["dist_to_powerup"] = float('inf')
            self.robot2["dist_to_powerup"] = float('inf')
            return

        # For simplicity, both robots target the first available powerup.
        # A more complex agent might benefit from individual targets.
        powerup_pos = self.powerups[0]
        self.robot1["dist_to_powerup"] = np.linalg.norm(self.robot1["pos"] - powerup_pos)
        self.robot2["dist_to_powerup"] = np.linalg.norm(self.robot2["pos"] - powerup_pos)


    # --- Rendering ---
    def _render_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Obstacles (Triangles)
        for pos in self.obstacles:
            points = [
                (pos[0], pos[1] - self.OBSTACLE_SIZE / 2),
                (pos[0] - self.OBSTACLE_SIZE / 2, pos[1] + self.OBSTACLE_SIZE / 2),
                (pos[0] + self.OBSTACLE_SIZE / 2, pos[1] + self.OBSTACLE_SIZE / 2),
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_OBSTACLE)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(p[0]), int(p[1])) for p in points], self.COLOR_OBSTACLE)

        # Powerups (Glowing Circles)
        for pos in self.powerups:
            self._draw_glow_circle(pos, self.COLOR_POWERUP, self.POWERUP_SIZE)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), int(p["radius"]))

        # Robots (Glowing Squares)
        self._draw_glow_square(self.robot1["pos"], self.COLOR_ROBOT1, self.ROBOT_SIZE)
        self._draw_glow_square(self.robot2["pos"], self.COLOR_ROBOT2, self.ROBOT_SIZE)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"POWER: {self.score}/{self.WIN_CONDITION}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Timer
        time_left = (self.MAX_STEPS - self.steps) / self.metadata["render_fps"]
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}s", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Energy Bars
        self._draw_energy_bar(self.robot1, self.COLOR_ROBOT1)
        self._draw_energy_bar(self.robot2, self.COLOR_ROBOT2)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "MISSION COMPLETE" if self.win else "SYSTEM FAILURE"
        text_surface = self.font_game_over.render(message, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text_surface, text_rect)

    def _draw_energy_bar(self, robot, color):
        pos = robot["pos"]
        energy_percent = robot["energy"] / self.INITIAL_ENERGY
        bar_width = 30
        bar_height = 5

        x = pos[0] - bar_width / 2
        y = pos[1] - self.ROBOT_SIZE - 10

        # Interpolate color from green to red
        bar_color = (
            self.COLOR_ENERGY_EMPTY[0] + (self.COLOR_ENERGY_FULL[0] - self.COLOR_ENERGY_EMPTY[0]) * energy_percent,
            self.COLOR_ENERGY_EMPTY[1] + (self.COLOR_ENERGY_FULL[1] - self.COLOR_ENERGY_EMPTY[1]) * energy_percent,
            self.COLOR_ENERGY_EMPTY[2] + (self.COLOR_ENERGY_FULL[2] - self.COLOR_ENERGY_EMPTY[2]) * energy_percent,
        )

        pygame.draw.rect(self.screen, (50, 50, 50), (int(x), int(y), bar_width, bar_height))
        pygame.draw.rect(self.screen, bar_color, (int(x), int(y), int(bar_width * energy_percent), bar_height))

    def _draw_glow_circle(self, pos, color, radius):
        x, y = int(pos[0]), int(pos[1])
        for i in range(radius, 0, -2):
            alpha = 128 * (1 - (i / radius))
            glow_color = (*color, int(alpha))
            pygame.gfxdraw.filled_circle(self.screen, x, y, i + 3, glow_color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, color)

    def _draw_glow_square(self, pos, color, size):
        x, y = int(pos[0] - size / 2), int(pos[1] - size / 2)
        rect = pygame.Rect(x, y, size, size)

        for i in range(4, 0, -1):
            glow_rect = rect.inflate(i*2, i*2)
            alpha = 100 * (1 - (i / 4))
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*color, int(alpha)), (0, 0, *glow_rect.size), border_radius=3)
            self.screen.blit(shape_surf, glow_rect.topleft)

        pygame.draw.rect(self.screen, color, rect, border_radius=2)

    # --- Particle System ---
    def _spawn_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            velocity = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(),
                "vel": velocity,
                "radius": self.np_random.uniform(1, 4),
                "lifespan": self.np_random.uniform(10, 20),
                "color": color
            })

    def _update_particles(self):
        for i in reversed(range(len(self.particles))):
            p = self.particles[i]
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["radius"] *= 0.95  # Shrink
            if p["lifespan"] <= 0 or p["radius"] < 0.5:
                self.particles.pop(i)

    def close(self):
        pygame.quit()

# --- Example Usage ---
if __name__ == "__main__":
    # To run with display, comment out the os.environ line at the top
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")

    # --- Manual Play ---
    # Controls:
    # Arrow Keys: Set movement direction for both robots
    # Space Bar: Move Robot 1 (Cyan)
    # Left Shift: Move Robot 2 (Magenta)
    # R: Reset environment

    obs, info = env.reset(seed=42)
    done = False

    # For display, we need to create a window
    use_display = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
    if use_display:
        pygame.display.set_caption("Dual Robot Energy Collector")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # [movement, move_r1, move_r2]

    running = True
    while running:
        if use_display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    done = False

            # Get key presses for manual control
            keys = pygame.key.get_pressed()

            # Movement direction
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            else:
                action[0] = 0

            # Robot movement triggers
            action[1] = 1 if keys[pygame.K_SPACE] else 0
            action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        else: # Simple auto-play for headless mode
            action = env.action_space.sample()


        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else: # If done, reset
            obs, info = env.reset()
            done = False


        # Render the observation to the display window
        if use_display:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            env.clock.tick(env.metadata["render_fps"])
        
        # Add a break for headless mode to prevent infinite loop
        if not use_display and env.steps > 2000:
            print("Headless run finished.")
            running = False


    env.close()