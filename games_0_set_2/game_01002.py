
# Generated: 2025-08-27T15:28:16.361298
# Source Brief: brief_01002.md
# Brief Index: 1002

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrows to move. Space to fire in your facing direction. Shift to rotate clockwise."
    )

    game_description = (
        "Control a robot in a grid-based arena, blasting enemies to achieve total robotic domination."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 10
    CELL_SIZE = 36
    ARENA_WIDTH = GRID_SIZE * CELL_SIZE
    ARENA_HEIGHT = GRID_SIZE * CELL_SIZE
    ARENA_X_OFFSET = (SCREEN_WIDTH - ARENA_WIDTH) // 2
    ARENA_Y_OFFSET = (SCREEN_HEIGHT - ARENA_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 45, 60)
    COLOR_WALL = (80, 90, 120)

    COLOR_ROBOT = (50, 150, 255)
    COLOR_ROBOT_ACCENT = (150, 220, 255)

    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_ACCENT = (255, 150, 150)

    COLOR_PLAYER_PROJ = (255, 255, 100)
    COLOR_ENEMY_PROJ = (255, 150, 50)

    COLOR_WHITE = (255, 255, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)

    # Game Parameters
    MAX_STEPS = 1000
    INITIAL_ENEMIES = 5
    ROBOT_MAX_HEALTH = 100
    ENEMY_MAX_HEALTH = 20
    PLAYER_PROJ_DAMAGE = 10
    ENEMY_PROJ_DAMAGE = 20
    PLAYER_FIRE_COOLDOWN = 2
    ENEMY_FIRE_RATE = 2 # An enemy fires on average every 2 steps
    HIT_FLASH_DURATION = 2

    # Directions (Up, Right, Down, Left)
    DIRECTIONS = [np.array([0, -1]), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)

        self.np_random = None
        self.robot = {}
        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_damage_flash = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_damage_flash = 0
        self.projectiles = []
        self.particles = []

        # Place robot
        self.robot = {
            "pos": self.np_random.integers(0, self.GRID_SIZE, size=2),
            "health": self.ROBOT_MAX_HEALTH,
            "facing": self.np_random.integers(0, 4), # 0:U, 1:R, 2:D, 3:L
            "fire_cooldown": 0,
            "hit_timer": 0,
        }

        # Spawn enemies
        self.enemies = []
        occupied_positions = {tuple(self.robot["pos"])}
        for _ in range(self.INITIAL_ENEMIES):
            while True:
                pos = self.np_random.integers(0, self.GRID_SIZE, size=2)
                if tuple(pos) not in occupied_positions:
                    occupied_positions.add(tuple(pos))
                    break
            self.enemies.append({
                "pos": pos,
                "health": self.ENEMY_MAX_HEALTH,
                "hit_timer": 0,
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.1  # Time penalty

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Update Cooldowns and Timers ---
        if self.robot["fire_cooldown"] > 0:
            self.robot["fire_cooldown"] -= 1
        if self.robot["hit_timer"] > 0:
            self.robot["hit_timer"] -= 1
        if self.player_damage_flash > 0:
            self.player_damage_flash -= 1
        for enemy in self.enemies:
            if enemy["hit_timer"] > 0:
                enemy["hit_timer"] -= 1

        # --- Player Actions ---
        if shift_held:
            self.robot["facing"] = (self.robot["facing"] + 1) % 4
        
        if movement > 0:
            direction_vec = self.DIRECTIONS[movement - 1]
            new_pos = self.robot["pos"] + direction_vec
            if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                self.robot["pos"] = new_pos

        if space_held and self.robot["fire_cooldown"] == 0:
            self._fire_projectile(self.robot, is_player=True)
            self.robot["fire_cooldown"] = self.PLAYER_FIRE_COOLDOWN

        # --- Enemy Actions ---
        for enemy in self.enemies:
            # Move
            move_dir_idx = self.np_random.integers(0, 5) # 0-3 for move, 4 for stay
            if move_dir_idx < 4:
                direction_vec = self.DIRECTIONS[move_dir_idx]
                new_pos = enemy["pos"] + direction_vec
                if 0 <= new_pos[0] < self.GRID_SIZE and 0 <= new_pos[1] < self.GRID_SIZE:
                    enemy["pos"] = new_pos
            # Fire
            if self.np_random.integers(0, self.ENEMY_FIRE_RATE) == 0:
                self._fire_projectile(enemy, is_player=False)

        # --- Update Projectiles & Collisions ---
        new_projectiles = []
        for proj in self.projectiles:
            proj["pos"] += proj["vel"]
            
            # Check wall collision
            if not (0 <= proj["pos"][0] < self.GRID_SIZE and 0 <= proj["pos"][1] < self.GRID_SIZE):
                self._spawn_sparks(proj["pos"] - proj["vel"], 5)
                continue # Projectile is destroyed
            
            collided = False
            if proj["is_player"]:
                for enemy in self.enemies:
                    if np.array_equal(proj["pos"], enemy["pos"]):
                        enemy["health"] -= self.PLAYER_PROJ_DAMAGE
                        enemy["hit_timer"] = self.HIT_FLASH_DURATION
                        reward += 1.0
                        collided = True
                        if enemy["health"] <= 0:
                            self._spawn_explosion(enemy["pos"], self.COLOR_ENEMY, 30)
                            reward += 2.0 # Kill bonus
                        else:
                            self._spawn_sparks(enemy["pos"], 8)
                        break
            else: # Enemy projectile
                if np.array_equal(proj["pos"], self.robot["pos"]):
                    self.robot["health"] -= self.ENEMY_PROJ_DAMAGE
                    self.robot["hit_timer"] = self.HIT_FLASH_DURATION
                    self.player_damage_flash = 3
                    reward -= 2.0
                    collided = True
                    self._spawn_sparks(self.robot["pos"], 12)
            
            if not collided:
                new_projectiles.append(proj)
        
        self.projectiles = new_projectiles
        self.enemies = [e for e in self.enemies if e["health"] > 0]
        
        # --- Update Particles ---
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

        # --- Check Termination ---
        terminated = False
        if self.robot["health"] <= 0:
            reward -= 100
            terminated = True
            self.game_over = True
            self.game_won = False
            self._spawn_explosion(self.robot["pos"], self.COLOR_ROBOT, 50)
        elif not self.enemies:
            reward += 100
            terminated = True
            self.game_over = True
            self.game_won = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _fire_projectile(self, owner, is_player):
        # sfx: laser_shoot
        start_pos = owner["pos"].copy()
        if is_player:
            direction_vec = self.DIRECTIONS[owner["facing"]]
        else: # Enemy aims at player
            delta = self.robot["pos"] - owner["pos"]
            if abs(delta[0]) > abs(delta[1]):
                direction_vec = np.array([np.sign(delta[0]), 0])
            elif delta[1] != 0:
                direction_vec = np.array([0, np.sign(delta[1])])
            else: # On same spot, fire randomly
                direction_vec = self.DIRECTIONS[self.np_random.integers(0, 4)]
        
        self.projectiles.append({
            "pos": start_pos,
            "vel": direction_vec,
            "is_player": is_player,
        })
        self._spawn_muzzle_flash(start_pos + direction_vec, direction_vec)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_particles()
        self._render_projectiles()
        self._render_enemies()
        self._render_robot()
        self._render_ui()

        if self.player_damage_flash > 0:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((255, 0, 0, 60))
            self.screen.blit(s, (0, 0))
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "robot_health": self.robot.get("health", 0),
            "enemies_remaining": len(self.enemies),
        }

    def _to_screen_coords(self, grid_pos):
        x = self.ARENA_X_OFFSET + grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.ARENA_Y_OFFSET + grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _render_grid(self):
        for i in range(self.GRID_SIZE + 1):
            # Vertical lines
            start_v = (self.ARENA_X_OFFSET + i * self.CELL_SIZE, self.ARENA_Y_OFFSET)
            end_v = (self.ARENA_X_OFFSET + i * self.CELL_SIZE, self.ARENA_Y_OFFSET + self.ARENA_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v)
            # Horizontal lines
            start_h = (self.ARENA_X_OFFSET, self.ARENA_Y_OFFSET + i * self.CELL_SIZE)
            end_h = (self.ARENA_X_OFFSET + self.ARENA_WIDTH, self.ARENA_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h)
        
        pygame.draw.rect(self.screen, self.COLOR_WALL, (self.ARENA_X_OFFSET, self.ARENA_Y_OFFSET, self.ARENA_WIDTH, self.ARENA_HEIGHT), 2)

    def _render_robot(self):
        if self.robot["health"] <= 0: return
        
        pos = self._to_screen_coords(self.robot["pos"])
        size = int(self.CELL_SIZE * 0.4)
        
        color = self.COLOR_WHITE if self.robot["hit_timer"] > 0 else self.COLOR_ROBOT
        accent_color = self.COLOR_WHITE if self.robot["hit_timer"] > 0 else self.COLOR_ROBOT_ACCENT

        pygame.draw.rect(self.screen, color, (pos[0] - size, pos[1] - size, size * 2, size * 2))
        pygame.draw.rect(self.screen, accent_color, (pos[0] - size, pos[1] - size, size * 2, size * 2), 2)

        # Facing indicator
        dir_vec = self.DIRECTIONS[self.robot["facing"]]
        p1 = pos
        p2 = (pos[0] + dir_vec[0] * size, pos[1] + dir_vec[1] * size)
        pygame.draw.line(self.screen, accent_color, p1, p2, 3)
        
        self._render_health_bar(pos, self.robot["health"], self.ROBOT_MAX_HEALTH)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = self._to_screen_coords(enemy["pos"])
            size = int(self.CELL_SIZE * 0.35)
            
            color = self.COLOR_WHITE if enemy["hit_timer"] > 0 else self.COLOR_ENEMY
            accent_color = self.COLOR_WHITE if enemy["hit_timer"] > 0 else self.COLOR_ENEMY_ACCENT

            pts = [
                (pos[0], pos[1] - size),
                (pos[0] + size, pos[1] + size),
                (pos[0] - size, pos[1] + size),
            ]
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, accent_color, pts, 2)
            
            self._render_health_bar(pos, enemy["health"], self.ENEMY_MAX_HEALTH)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos = self._to_screen_coords(proj["pos"])
            color = self.COLOR_PLAYER_PROJ if proj["is_player"] else self.COLOR_ENEMY_PROJ
            size = self.CELL_SIZE // 6
            pygame.draw.rect(self.screen, color, (pos[0] - size, pos[1] - size, size * 2, size * 2))

    def _render_health_bar(self, pos, current, maximum):
        bar_width = self.CELL_SIZE * 0.8
        bar_height = 5
        y_offset = self.CELL_SIZE * 0.6
        
        health_pct = max(0, current / maximum)
        
        bg_rect = pygame.Rect(pos[0] - bar_width / 2, pos[1] + y_offset, bar_width, bar_height)
        fill_rect = pygame.Rect(pos[0] - bar_width / 2, pos[1] + y_offset, bar_width * health_pct, bar_height)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, bg_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fill_rect)
        pygame.draw.rect(self.screen, self.COLOR_WHITE, bg_rect, 1)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = p["color"] + (alpha,)
            pos = self._to_screen_coords(p["pos"])
            size = max(1, int(p["size"] * (p["life"] / p["max_life"])))
            
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (size, size), size)
            self.screen.blit(s, (pos[0]-size, pos[1]-size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        health_text = self.font_ui.render(f"HEALTH: {max(0, self.robot['health'])}", True, self.COLOR_UI_TEXT)
        enemies_text = self.font_ui.render(f"ENEMIES: {len(self.enemies)}", True, self.COLOR_UI_TEXT)
        
        self.screen.blit(health_text, (10, 10))
        self.screen.blit(enemies_text, (self.SCREEN_WIDTH - enemies_text.get_width() - 10, 10))

    def _render_game_over(self):
        text_str = "VICTORY" if self.game_won else "GAME OVER"
        color = self.COLOR_ROBOT_ACCENT if self.game_won else self.COLOR_ENEMY
        
        text_surf = self.font_game_over.render(text_str, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        
        # Draw a semi-transparent background for readability
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 180))
        self.screen.blit(overlay, (0, 0))
        
        self.screen.blit(text_surf, text_rect)

    def _spawn_explosion(self, pos, base_color, count):
        # sfx: explosion
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.05, 0.2)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.astype(float) + self.np_random.uniform(-0.2, 0.2, size=2),
                "vel": vel,
                "life": self.np_random.integers(15, 30),
                "max_life": 30,
                "color": random.choice([base_color, (255, 255, 100), self.COLOR_WHITE]),
                "size": self.np_random.integers(5, 12),
            })

    def _spawn_sparks(self, pos, count):
        # sfx: hit_spark
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.1, 0.3)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": pos.astype(float),
                "vel": vel,
                "life": self.np_random.integers(5, 10),
                "max_life": 10,
                "color": self.COLOR_WHITE,
                "size": self.np_random.integers(2, 4),
            })

    def _spawn_muzzle_flash(self, pos, direction):
        # sfx: muzzle_flash
        for i in range(3):
            self.particles.append({
                "pos": pos.astype(float) - direction * i * 0.1,
                "vel": direction * 0.05,
                "life": 2,
                "max_life": 2,
                "color": self.COLOR_PLAYER_PROJ,
                "size": 8 - i * 2,
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not be executed when the environment is used by an RL agent
    
    # Set this to False to run headless (for RL training)
    human_playing = True
    
    if not human_playing:
        # Test headless run
        env = GameEnv()
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print("Headless episode finished.")
                env.reset()
        env.close()
        print("Headless test complete.")
    else:
        # Manual play mode
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        # Re-initialize pygame with a display
        pygame.display.init()
        pygame.display.set_caption("Robo-Grid Blast")
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        running = True
        while running:
            movement = 0 # none
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1

            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Display the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            if terminated:
                print(f"Game Over! Final Score: {info['score']}")
                pygame.time.wait(2000) # Pause for 2 seconds
                obs, info = env.reset()

            env.clock.tick(10) # Run at 10 steps per second for playability

        env.close()