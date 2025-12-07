
# Generated: 2025-08-28T03:05:39.786111
# Source Brief: brief_01915.md
# Brief Index: 1915

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    """
    A top-down, grid-based arcade shooter where the player controls a robot
    to destroy all enemies in an arena. The game is turn-based, with each
    action advancing the game state by one step.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Use arrow keys to move one cell. Press Space to fire. Press Shift to rotate your turret."
    )
    game_description = (
        "Tactical grid-based shooter. Annihilate all enemy bots before they overwhelm you. Every move counts!"
    )

    # Game configuration
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 20
        self.GRID_W = self.WIDTH // self.GRID_SIZE
        self.GRID_H = self.HEIGHT // self.GRID_SIZE
        self.MAX_STEPS = 1000
        self.INITIAL_ENEMIES = 30
        self.ROBOT_MAX_HEALTH = 50

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_ROBOT = (60, 120, 255)
        self.COLOR_ROBOT_TURRET = (150, 200, 255)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_PROJECTILE = (255, 255, 100)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_HEALTH_FG = (80, 220, 80)
        self.COLOR_HEALTH_BG = (100, 40, 40)
        self.COLOR_WIN = (100, 255, 100)
        self.COLOR_LOSE = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.robot_pos = (0, 0)
        self.robot_health = 0
        self.robot_facing = 0  # 0:Up, 1:Right, 2:Down, 3:Left

        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.rng = np.random.default_rng()

        # Initialize state variables
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        # --- Robot State ---
        self.robot_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.robot_health = self.ROBOT_MAX_HEALTH
        self.robot_facing = 1  # Start facing right

        # --- Entity Lists ---
        self.enemies = []
        self.projectiles = []
        self.particles = []

        # --- Spawn Enemies ---
        occupied_positions = {self.robot_pos}
        for _ in range(self.INITIAL_ENEMIES):
            while True:
                pos = (self.rng.integers(0, self.GRID_W), self.rng.integers(0, self.GRID_H))
                if pos not in occupied_positions:
                    self.enemies.append({"pos": pos})
                    occupied_positions.add(pos)
                    break
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Time penalty

        # --- 1. Process Player Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if shift_held:
            self.robot_facing = (self.robot_facing + 1) % 4
            # SFX: Turret rotate sound
            self._create_particles(self.robot_pos, self.COLOR_ROBOT_TURRET, 5, 2)

        if space_held:
            self._fire_projectile()
            # SFX: Laser fire sound

        move_dir = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement)
        if move_dir:
            next_pos = (self.robot_pos[0] + move_dir[0], self.robot_pos[1] + move_dir[1])
            if 0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H:
                self.robot_pos = next_pos

        # --- 2. Update Game Entities ---
        # Move projectiles and check for hits
        projectiles_to_keep = []
        for p in self.projectiles:
            p["pos"] = (p["pos"][0] + p["dir"][0], p["pos"][1] + p["dir"][1])
            
            hit_enemy = None
            for enemy in self.enemies:
                if p["pos"] == enemy["pos"]:
                    hit_enemy = enemy
                    break

            if hit_enemy:
                self.enemies.remove(hit_enemy)
                self.score += 1
                reward += 10
                # SFX: Explosion sound
                self._create_particles(p["pos"], self.COLOR_ENEMY, 20, 4)
            elif 0 <= p["pos"][0] < self.GRID_W and 0 <= p["pos"][1] < self.GRID_H:
                projectiles_to_keep.append(p)
            else:
                # SFX: Projectile fizzle sound
                self._create_particles(p["pos"], self.COLOR_PROJECTILE, 3, 1)

        self.projectiles = projectiles_to_keep

        # Move enemies
        for enemy in self.enemies:
            move = self.rng.choice([(0,0), (0,1), (0,-1), (1,0), (-1,0)])
            next_pos = (enemy["pos"][0] + move[0], enemy["pos"][1] + move[1])
            if 0 <= next_pos[0] < self.GRID_W and 0 <= next_pos[1] < self.GRID_H:
                enemy["pos"] = next_pos

        # --- 3. Resolve Collisions & State Changes ---
        # Check for robot-enemy collisions
        for enemy in self.enemies:
            if self.robot_pos == enemy["pos"]:
                self.robot_health -= 1
                reward -= 1
                # SFX: Damage taken sound
                self._create_particles(self.robot_pos, self.COLOR_ROBOT, 15, 3)
                if self.robot_health < 0: self.robot_health = 0

        # --- 4. Check for Termination ---
        terminated = False
        if self.robot_health <= 0:
            terminated = True
            self.game_over = True
            self.win_condition = False
            reward -= 100
        elif not self.enemies:
            terminated = True
            self.game_over = True
            self.win_condition = True
            reward += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            self.win_condition = False

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _fire_projectile(self):
        direction_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        proj_dir = direction_map[self.robot_facing]
        start_pos = (self.robot_pos[0] + proj_dir[0], self.robot_pos[1] + proj_dir[1])
        
        self.projectiles.append({"pos": start_pos, "dir": proj_dir})
        self._create_particles(self.robot_pos, self.COLOR_PROJECTILE, 8, 3)

    def _get_observation(self):
        # --- Background ---
        self.screen.fill(self.COLOR_BG)
        self._draw_grid()

        # --- Particles ---
        self._update_and_draw_particles()

        # --- Game Entities ---
        self._draw_enemies()
        self._draw_projectiles()
        self._draw_robot()

        # --- UI Overlay ---
        self._draw_ui()

        # --- Game Over Message ---
        if self.game_over:
            self._draw_game_over_message()

        # --- Convert to numpy array ---
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.robot_health,
            "enemies_left": len(self.enemies),
        }

    # --- Helper Rendering Methods ---

    def _grid_to_pixel(self, grid_pos):
        return (
            int(grid_pos[0] * self.GRID_SIZE + self.GRID_SIZE / 2),
            int(grid_pos[1] * self.GRID_SIZE + self.GRID_SIZE / 2)
        )

    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.GRID_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_robot(self):
        px, py = self._grid_to_pixel(self.robot_pos)
        size = int(self.GRID_SIZE * 0.8)
        rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
        pygame.draw.rect(self.screen, self.COLOR_ROBOT, rect, border_radius=3)

        # Draw turret indicating facing direction
        direction_vectors = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
        dx, dy = direction_vectors[self.robot_facing]
        
        turret_len = self.GRID_SIZE * 0.6
        turret_end_x = px + dx * turret_len
        turret_end_y = py + dy * turret_len
        
        pygame.draw.line(self.screen, self.COLOR_ROBOT_TURRET, (px, py), (turret_end_x, turret_end_y), 4)
        pygame.gfxdraw.aacircle(self.screen, px, py, 4, self.COLOR_ROBOT_TURRET)
        pygame.gfxdraw.filled_circle(self.screen, px, py, 4, self.COLOR_ROBOT_TURRET)

    def _draw_enemies(self):
        size = int(self.GRID_SIZE * 0.7)
        for enemy in self.enemies:
            px, py = self._grid_to_pixel(enemy["pos"])
            rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, rect, border_radius=2)

    def _draw_projectiles(self):
        for p in self.projectiles:
            px, py = self._grid_to_pixel(p["pos"])
            size = int(self.GRID_SIZE * 0.3)
            rect = pygame.Rect(px - size // 2, py - size // 2, size, size)
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, rect)

    def _draw_ui(self):
        # Health Bar
        health_ratio = self.robot_health / self.ROBOT_MAX_HEALTH
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (10, 10, max(0, bar_width * health_ratio), bar_height))
        
        health_text = self.font_ui.render(f"HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 11))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

    def _draw_game_over_message(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.win_condition:
            msg_text = "SYSTEM CLEAR"
            msg_color = self.COLOR_WIN
        else:
            msg_text = "SYSTEM FAILURE"
            msg_color = self.COLOR_LOSE
            
        text_surf = self.font_msg.render(msg_text, True, msg_color)
        text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    # --- Particle System ---
    def _create_particles(self, grid_pos, color, count, max_speed):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(count):
            self.particles.append({
                "pos": [px, py],
                "vel": [(self.rng.random() - 0.5) * max_speed, (self.rng.random() - 0.5) * max_speed],
                "life": self.rng.integers(10, 20),
                "color": color,
            })

    def _update_and_draw_particles(self):
        particles_to_keep = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            if p["life"] > 0:
                particles_to_keep.append(p)
                radius = int(p["life"] * 0.2)
                if radius > 0:
                    pos_int = (int(p["pos"][0]), int(p["pos"][1]))
                    pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, p["color"])
        self.particles = particles_to_keep

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play Example ---
    # This requires a display and is for testing/demonstration.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "dummy" # Ensure it runs headless
        pygame.display.init()
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Grid Bot Arena")
        os.environ["SDL_VIDEODRIVER"] = "" # Reset for display
    except pygame.error:
        print("Pygame display not available. Running in headless mode.")
        screen = None

    obs, info = env.reset()
    done = False
    
    if screen:
        running = True
        while running:
            action = [0, 0, 0] # Default action: no-op
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        obs, info = env.reset()
                        done = False

            if not done:
                keys = pygame.key.get_pressed()
                movement = 0
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                space = 1 if keys[pygame.K_SPACE] else 0
                shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                
                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Render the observation to the display window
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # Since auto_advance is False, we control the step rate
            pygame.time.wait(100) # 10 steps per second for manual play

        env.close()
    else:
        # --- Standard RL Loop Example ---
        print("\nRunning a short headless episode...")
        obs, info = env.reset()
        total_reward = 0
        for i in range(env.MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if i % 100 == 0:
                print(f"Step {i}: Reward={reward:.2f}, Info={info}")
            if terminated or truncated:
                print(f"Episode finished after {i+1} steps. Final Info: {info}, Total Reward: {total_reward:.2f}")
                break
        env.close()