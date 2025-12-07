import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:37:38.116895
# Source Brief: brief_01177.md
# Brief Index: 1177
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "A futuristic racing game where you match your vehicle's color to the track for speed boosts. "
        "Fire projectiles to terraform the track ahead and create your own path."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to steer and accelerate/brake. Press space to cycle your vehicle's color "
        "and shift to fire a terraforming projectile."
    )
    auto_advance = True

    # --- Class-level variables for state that persists across resets ---
    unlocked_colors = [0, 1, 2] # R, G, B
    target_lap_time = 60.0
    successful_laps_counter = 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 2500 # Increased for longer race potential

        # --- Colors ---
        self.COLOR_BG = (15, 10, 40)
        self.COLOR_NEUTRAL = (50, 50, 60)
        self.COLOR_OBSTACLE = (10, 5, 20)
        self.TERRAIN_COLORS = [
            (255, 50, 50),   # 0: Red
            (50, 255, 50),   # 1: Green
            (50, 100, 255),  # 2: Blue
            (255, 255, 0),   # 3: Yellow (Unlockable)
            (0, 255, 255),   # 4: Cyan (Unlockable)
            (255, 0, 255),   # 5: Magenta (Unlockable)
        ]
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_UI_ACCENT = (100, 150, 255)
        
        # --- Action & Observation Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lap_time = 0.0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0.0
        self.player_color_index = 0
        self.projectiles = []
        self.particles = []
        self.speed_lines = []
        self.last_space_held = False
        self.last_shift_held = False
        self.fire_cooldown = 0
        self.terrain_grid = None
        self.track_centerline = []
        self.checkpoints_passed = set()

        # --- Physics Constants ---
        self.TURN_SPEED = 5.0
        self.ACCELERATION = 0.4
        self.BRAKE_FORCE = 0.6
        self.FRICTION = 0.96
        self.MAX_SPEED_NORMAL = 5.0
        self.MAX_SPEED_BOOST = 9.0
        self.MAX_SPEED_PENALTY = 2.5
        self.PROJECTILE_SPEED = 15.0
        self.PROJECTILE_LIFETIME = 20 # steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.lap_time = 0.0
        
        self._generate_track()
        
        self.player_pos = pygame.Vector2(100, self.track_centerline[10])
        self.player_vel = pygame.Vector2(0, 0)
        self.player_angle = 0.0
        self.player_color_index = 0
        
        self.projectiles = []
        self.particles = deque(maxlen=200)
        self.speed_lines = deque(maxlen=50)
        
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True
        self.fire_cooldown = 0
        self.checkpoints_passed = set()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.lap_time += 1 / self.FPS
        
        # --- Handle Actions ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        # --- Update Game Logic ---
        self._update_player()
        self._update_projectiles()
        self._update_particles()
        self._update_speed_lines()

        # --- Check State & Calculate Reward ---
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _generate_track(self):
        self.TILE_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.TILE_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.TILE_SIZE
        self.terrain_grid = np.full((self.GRID_WIDTH, self.GRID_HEIGHT), -1, dtype=int) # -1 is obstacle
        self.track_centerline = []
        
        track_width_tiles = 6
        amp = self.GRID_HEIGHT / 4
        freq = self.np_random.uniform(1.5, 2.5)
        
        # Generate centerline
        for i in range(self.GRID_WIDTH):
            y_center = self.HEIGHT / 2 + amp * math.sin(freq * i / self.GRID_WIDTH * 2 * math.pi)
            self.track_centerline.append(y_center)
            
            # "Paint" the track onto the grid
            y_tile_center = int(y_center / self.TILE_SIZE)
            for w in range(-track_width_tiles // 2, track_width_tiles // 2):
                y = y_tile_center + w
                if 0 <= y < self.GRID_HEIGHT:
                    # Assign colors in segments
                    segment = (i // (self.GRID_WIDTH // 4)) % 3
                    self.terrain_grid[i, y] = segment

    def _handle_input(self, movement, space_held, shift_held):
        # Movement
        if movement == 1: # Up
            self.player_vel += pygame.Vector2(self.ACCELERATION, 0).rotate(-self.player_angle)
            # sfx: engine_accelerate
        elif movement == 2: # Down
            self.player_vel *= 0.9 # More effective brake
            # sfx: brake_screech
        if movement == 3: # Left
            self.player_angle -= self.TURN_SPEED
        elif movement == 4: # Right
            self.player_angle += self.TURN_SPEED
        
        # Color Switch (on button press)
        if space_held and not self.last_space_held:
            self.player_color_index = (self.player_color_index + 1) % len(self.unlocked_colors)
            # sfx: color_switch
            self._create_particles(self.player_pos, 15, self.TERRAIN_COLORS[self.unlocked_colors[self.player_color_index]])
        self.last_space_held = space_held
        
        # Fire Projectile (on button press with cooldown)
        self.fire_cooldown = max(0, self.fire_cooldown - 1)
        if shift_held and not self.last_shift_held and self.fire_cooldown == 0:
            self.fire_cooldown = 10 # 1/3 second cooldown
            direction = pygame.Vector2(1, 0).rotate(-self.player_angle)
            start_pos = self.player_pos + direction * 15
            self.projectiles.append({
                "pos": start_pos,
                "vel": direction * self.PROJECTILE_SPEED,
                "color": self.unlocked_colors[self.player_color_index],
                "lifetime": self.PROJECTILE_LIFETIME
            })
            # sfx: projectile_fire
        self.last_shift_held = shift_held

    def _update_player(self):
        current_max_speed = self.MAX_SPEED_NORMAL
        
        # Terrain interaction
        tile_x, tile_y = int(self.player_pos.x / self.TILE_SIZE), int(self.player_pos.y / self.TILE_SIZE)
        if 0 <= tile_x < self.GRID_WIDTH and 0 <= tile_y < self.GRID_HEIGHT:
            terrain_color_code = self.terrain_grid[tile_x, tile_y]
            player_color_code = self.unlocked_colors[self.player_color_index]

            if terrain_color_code == player_color_code:
                current_max_speed = self.MAX_SPEED_BOOST
                if self.player_vel.length() > 2.0:
                    self._create_particles(self.player_pos - self.player_vel.normalize()*10, 1, self.COLOR_WHITE, 0.5, 10)
            elif terrain_color_code != -1: # Not an obstacle, but mismatch
                current_max_speed = self.MAX_SPEED_PENALTY
        
        # Physics
        if self.player_vel.length() > current_max_speed:
            self.player_vel.scale_to_length(current_max_speed)
        
        self.player_vel *= self.FRICTION
        self.player_pos += self.player_vel
    
    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p["pos"] += p["vel"]
            p["lifetime"] -= 1
            
            tile_x, tile_y = int(p["pos"].x / self.TILE_SIZE), int(p["pos"].y / self.TILE_SIZE)
            
            hit = False
            if 0 <= tile_x < self.GRID_WIDTH and 0 <= tile_y < self.GRID_HEIGHT:
                if self.terrain_grid[tile_x, tile_y] != -1:
                    # Terraforming
                    self.terrain_grid[tile_x, tile_y] = p["color"]
                    hit = True
                    # sfx: terraform_impact
            
            if p["lifetime"] <= 0 or hit:
                self._create_particles(p["pos"], 20, self.TERRAIN_COLORS[p["color"]])
                self.projectiles.remove(p)

    def _create_particles(self, pos, count, color, speed_mult=1.0, lifetime=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 360)
            speed = self.np_random.uniform(1, 4) * speed_mult
            vel = pygame.Vector2(speed, 0).rotate(angle)
            self.particles.append({"pos": pos.copy(), "vel": vel, "lifetime": lifetime, "color": color})

    def _update_particles(self):
        for p in list(self.particles):
            p["pos"] += p["vel"]
            p["vel"] *= 0.9
            p["lifetime"] -= 1
            if p["lifetime"] <= 0:
                self.particles.remove(p)

    def _update_speed_lines(self):
        speed_ratio = self.player_vel.length() / self.MAX_SPEED_BOOST
        if speed_ratio > 0.7 and self.np_random.random() < speed_ratio:
            start_pos = pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
            direction = (start_pos - pygame.Vector2(self.WIDTH/2, self.HEIGHT/2)).normalize()
            self.speed_lines.append({
                "pos": start_pos,
                "dir": direction,
                "len": self.np_random.uniform(20, 50) * speed_ratio,
                "speed": self.np_random.uniform(10, 20) * speed_ratio,
                "lifetime": 10
            })
        
        for sl in list(self.speed_lines):
            sl["pos"] += sl["dir"] * sl["speed"]
            sl["lifetime"] -= 1
            if sl["lifetime"] <= 0:
                self.speed_lines.remove(sl)

    def _calculate_reward(self):
        reward = 0.0
        
        # Continuous rewards
        tile_x, tile_y = int(self.player_pos.x / self.TILE_SIZE), int(self.player_pos.y / self.TILE_SIZE)
        if 0 <= tile_x < self.GRID_WIDTH and 0 <= tile_y < self.GRID_HEIGHT:
            terrain_color_code = self.terrain_grid[tile_x, tile_y]
            player_color_code = self.unlocked_colors[self.player_color_index]
            
            if terrain_color_code == player_color_code:
                reward += 0.1 # Color match boost
            elif terrain_color_code != -1:
                reward -= 0.1 # Color mismatch penalty
        
        if self.player_vel.length() > 1.0:
            reward += 0.01 # Forward movement
            
        return reward

    def _check_termination(self):
        # Crash condition
        tile_x, tile_y = int(self.player_pos.x / self.TILE_SIZE), int(self.player_pos.y / self.TILE_SIZE)
        crashed = not (0 <= tile_x < self.GRID_WIDTH and 0 <= tile_y < self.GRID_HEIGHT and self.terrain_grid[tile_x, tile_y] != -1)
        
        if crashed:
            self.score -= 100
            # sfx: crash_explosion
            self._create_particles(self.player_pos, 50, self.COLOR_WHITE)
            return True
            
        # Lap completion
        start_finish_x = 100
        checkpoint_x = self.WIDTH / 2
        
        if self.player_pos.x > checkpoint_x:
            self.checkpoints_passed.add(1)
            
        if self.player_pos.x < start_finish_x and 1 in self.checkpoints_passed:
            self.score += 5 # Base reward for finishing
            if self.lap_time < self.target_lap_time:
                self.score += 100 # Bonus for beating target
                # sfx: lap_complete_success
                GameEnv.successful_laps_counter += 1
                if GameEnv.successful_laps_counter % 5 == 0:
                    GameEnv.target_lap_time = max(15.0, GameEnv.target_lap_time - 1.0)
                
                # Unlock progression
                if self.lap_time < 30 and 3 not in GameEnv.unlocked_colors: GameEnv.unlocked_colors.append(3)
                if self.lap_time < 25 and 4 not in GameEnv.unlocked_colors: GameEnv.unlocked_colors.append(4)
                if self.lap_time < 20 and 5 not in GameEnv.unlocked_colors: GameEnv.unlocked_colors.append(5)
            else:
                # sfx: lap_complete_fail
                pass

            return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap_time": self.lap_time,
            "target_lap_time": self.target_lap_time
        }

    def _render_game(self):
        # --- Camera offset to center player ---
        cam_x = self.player_pos.x - self.WIDTH / 2
        cam_y = self.player_pos.y - self.HEIGHT / 2

        # --- Draw Terrain ---
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                color_code = self.terrain_grid[x, y]
                if color_code != -1:
                    rect = pygame.Rect(x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y, self.TILE_SIZE, self.TILE_SIZE)
                    if rect.colliderect(self.screen.get_rect()): # Culling
                        pygame.draw.rect(self.screen, self.TERRAIN_COLORS[color_code], rect)
                else: # Obstacle/Boundary
                    rect = pygame.Rect(x * self.TILE_SIZE - cam_x, y * self.TILE_SIZE - cam_y, self.TILE_SIZE, self.TILE_SIZE)
                    if rect.colliderect(self.screen.get_rect()):
                        pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
        
        # --- Draw Start/Finish Line ---
        start_finish_x = 100 - cam_x
        for y in range(0, self.HEIGHT, 20):
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (start_finish_x, y, 5, 10))
            pygame.draw.rect(self.screen, self.COLOR_WHITE, (start_finish_x + 5, y + 10, 5, 10))

        # --- Draw Speed Lines ---
        for sl in self.speed_lines:
            alpha = int(255 * (sl['lifetime'] / 10))
            end_pos = sl['pos'] + sl['dir'] * sl['len']
            # We can't use alpha with standard draw, so we skip it for simplicity
            # For a real game, you'd use a separate surface with per-pixel alpha.
            pygame.draw.aaline(self.screen, self.COLOR_WHITE, sl['pos'], end_pos)

        # --- Draw Particles ---
        for p in self.particles:
            alpha = max(0, min(255, int(255 * p["lifetime"] / 20)))
            color = p["color"] + (alpha,)
            size = max(1, int(p["lifetime"] / 4))
            # Pygame's GFX draw doesn't support alpha in its main circle func
            # We simulate it with a solid circle
            pygame.gfxdraw.filled_circle(self.screen, int(p["pos"].x - cam_x), int(p["pos"].y - cam_y), size, color[:3])

        # --- Draw Projectiles ---
        for p in self.projectiles:
            pos = p["pos"] - pygame.Vector2(cam_x, cam_y)
            color = self.TERRAIN_COLORS[p["color"]]
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 5, color)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 5, self.COLOR_WHITE)

        # --- Draw Player ---
        if not self.game_over:
            player_screen_pos = self.player_pos - pygame.Vector2(cam_x, cam_y)
            player_color = self.TERRAIN_COLORS[self.unlocked_colors[self.player_color_index]]
            
            # Glow effect
            glow_radius = 20
            glow_color = player_color
            s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, glow_color + (30,), (glow_radius, glow_radius), glow_radius)
            pygame.draw.circle(s, glow_color + (50,), (glow_radius, glow_radius), int(glow_radius * 0.7))
            self.screen.blit(s, (player_screen_pos.x - glow_radius, player_screen_pos.y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

            # Vehicle shape
            size = 12
            p1 = player_screen_pos + pygame.Vector2(size, 0).rotate(-self.player_angle)
            p2 = player_screen_pos + pygame.Vector2(-size/2, -size/1.5).rotate(-self.player_angle)
            p3 = player_screen_pos + pygame.Vector2(-size/2, size/1.5).rotate(-self.player_angle)
            points = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y)]
            
            pygame.gfxdraw.aapolygon(self.screen, points, player_color)
            pygame.gfxdraw.filled_polygon(self.screen, points, player_color)
            
            # Core
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_WHITE)

    def _render_ui(self):
        # Speedometer
        speed = self.player_vel.length() * 10
        speed_text = self.font_large.render(f"{speed:.0f} KPH", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (20, 20))

        # Lap Timer
        time_text = self.font_small.render(f"TIME: {self.lap_time:.2f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (20, 50))

        # Target Time
        target_text = self.font_small.render(f"TARGET: {self.target_lap_time:.2f}", True, self.COLOR_UI_ACCENT)
        self.screen.blit(target_text, (20, 70))
        
        # Current Color Indicator
        color_label = self.font_small.render("COLOR:", True, self.COLOR_UI_TEXT)
        self.screen.blit(color_label, (20, 100))
        player_color = self.TERRAIN_COLORS[self.unlocked_colors[self.player_color_index]]
        pygame.draw.rect(self.screen, player_color, (80, 100, 20, 15))
        pygame.draw.rect(self.screen, self.COLOR_WHITE, (80, 100, 20, 15), 1)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    # This validation logic was in the original __init__, moved here for clarity
    def validate_implementation(env):
        print("Validating implementation...")
        # Test action space
        assert env.action_space.shape == (3,)
        assert env.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs, _ = env.reset()
        assert test_obs.shape == (env.HEIGHT, env.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (env.HEIGHT, env.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(test_action)
        assert obs.shape == (env.HEIGHT, env.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    env = GameEnv()
    validate_implementation(env)
    
    # --- Manual Play ---
    # To run manual play, you need a display.
    # If you run this script with a display, comment out the `os.environ` line at the top.
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Terraform Racer")
        clock = pygame.time.Clock()
        
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        
        while not terminated:
            movement = 0 # No-op
            space = 0
            shift = 0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

            action = [movement, space, shift]
            
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            if term or trunc:
                print(f"Episode Finished. Final Score: {info['score']:.2f}, Lap Time: {info['lap_time']:.2f}")
                # Reset for another round
                obs, info = env.reset()
                if term or trunc: # if episode ended, break loop
                    terminated = True

            # Transpose back for pygame display
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            clock.tick(env.FPS)
            
        env.close()
    except pygame.error as e:
        print(f"Pygame display error: {e}")
        print("Manual play requires a display. If you're in a headless environment, this is expected.")