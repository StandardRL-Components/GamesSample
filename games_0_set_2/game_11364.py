import gymnasium as gym
import os
import pygame
import numpy as np
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame.init()
pygame.font.init()

# The following import is conditional to avoid errors if gfxdraw is not available.
try:
    import pygame.gfxdraw
except ImportError:
    print("pygame.gfxdraw not available, some graphics will be simplified.")
    pygame.gfxdraw = None


class GameEnv(gym.Env):
    """
    Artillery Game Gymnasium Environment.

    **Objective:** Hit the moving target 3 times before running out of 5 projectiles.
    **Mechanics:**
    - Adjust launch angle and power.
    - Launch a projectile affected by gravity and wind.
    - After a successful hit, the next projectile becomes a guided missile.
    - The wind changes direction and strength periodically.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    game_description = (
        "Fire your cannon to hit the moving target, adjusting your angle and power to account for wind. "
        "Hitting the target rewards you with a guided missile for your next shot."
    )
    user_guide = (
        "Controls: Use ←→ arrow keys to aim the cannon and ↑↓ to adjust power. Press space to fire."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PLATFORM = (60, 60, 70)
    COLOR_LAUNCHER = (180, 180, 190)
    COLOR_PROJECTILE = (255, 80, 80)
    COLOR_GUIDED_PROJECTILE = (255, 150, 50)
    COLOR_TARGET = (80, 255, 80)
    COLOR_WIND = (100, 150, 255, 150)
    COLOR_TRAIL = (200, 120, 120)
    COLOR_TEXT = (220, 220, 220)
    COLOR_POWER_BAR_BG = (40, 40, 50)
    COLOR_POWER_BAR_FILL = (255, 200, 0)
    COLOR_PREDICTION = (100, 100, 110, 100)

    # Physics & Game Rules
    GRAVITY = 0.15
    MAX_AMMO = 5
    HITS_TO_WIN = 3
    MAX_STEPS = 1000
    WIND_CHANGE_INTERVAL = 60 # Steps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State Variables (initialized in reset) ---
        self.steps = None
        self.score = None
        self.game_over = None
        self.launcher_pos = None
        self.launch_angle = None
        self.launch_power = None
        self.projectiles = None
        self.particles = None
        self.target_pos = None
        self.target_phase = None
        self.target_hits = None
        self.remaining_ammo = None
        self.wind_vector = None
        self.is_guided_missile_active = None
        self.previous_space_held = None
        self.last_launch_min_dist = None
        self.current_min_dist_to_target = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self.launcher_pos = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30)
        self.launch_angle = -90.0  # Straight up
        self.launch_power = 5.0
        
        self.projectiles = []
        self.particles = []
        
        self.target_phase = self.np_random.uniform(0, 2 * math.pi)
        self._update_target()
        
        self.target_hits = 0
        self.remaining_ammo = self.MAX_AMMO
        
        self.wind_vector = (0, 0)
        self._update_wind()
        
        self.is_guided_missile_active = False
        self.previous_space_held = False

        # For proximity reward calculation
        self.last_launch_min_dist = float('inf')
        self.current_min_dist_to_target = float('inf')

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0.0

        # --- Handle Input ---
        if not self.projectiles: # Can only aim when no projectile is in flight
            # Adjust angle
            if movement == 3: self.launch_angle -= 1.5
            if movement == 4: self.launch_angle += 1.5
            self.launch_angle = np.clip(self.launch_angle, -160, -20)
            
            # Adjust power
            if movement == 1: self.launch_power += 0.2
            if movement == 2: self.launch_power -= 0.2
            self.launch_power = np.clip(self.launch_power, 3.0, 15.0)

            # Launch on spacebar press (rising edge)
            if space_held and not self.previous_space_held and self.remaining_ammo > 0:
                self._launch_projectile()

        self.previous_space_held = space_held
        
        # --- Update Game State ---
        self.steps += 1
        self._update_target()
        
        if self.steps % self.WIND_CHANGE_INTERVAL == 0:
            self._update_wind()

        reward += self._update_projectiles()
        self._update_particles()
        
        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.target_hits >= self.HITS_TO_WIN:
                reward += 50.0 # Win bonus
            else:
                reward -= 50.0 # Loss penalty

        self.score += reward
        truncated = self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _launch_projectile(self):
        self.remaining_ammo -= 1
        self.current_min_dist_to_target = float('inf')
        
        angle_rad = math.radians(self.launch_angle)
        vel_x = self.launch_power * math.cos(angle_rad)
        vel_y = self.launch_power * math.sin(angle_rad)
        
        proj_type = 'guided' if self.is_guided_missile_active else 'normal'
        self.projectiles.append({
            'pos': list(self.launcher_pos),
            'vel': [vel_x, vel_y],
            'type': proj_type,
            'trail': []
        })
        self.is_guided_missile_active = False # Consume the guided missile status

    def _update_target(self):
        y_pos = 50
        x_range = self.SCREEN_WIDTH - 100
        x_offset = 50
        self.target_pos = (
            x_offset + x_range / 2 * (1 + math.sin(self.target_phase + self.steps * 0.02)),
            y_pos
        )

    def _update_wind(self):
        wind_angle = self.np_random.uniform(-math.pi / 4, math.pi / 4) # +/- 45 degrees
        wind_strength = self.np_random.uniform(0.01, 0.05)
        self.wind_vector = (
            wind_strength * math.cos(wind_angle),
            wind_strength * math.sin(wind_angle)
        )

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []

        for i, p in enumerate(self.projectiles):
            # Update trail
            p['trail'].append(tuple(p['pos']))
            if len(p['trail']) > 30:
                p['trail'].pop(0)
            
            # Homing for guided missiles
            if p['type'] == 'guided':
                dir_to_target = (self.target_pos[0] - p['pos'][0], self.target_pos[1] - p['pos'][1])
                dist = math.hypot(*dir_to_target)
                if dist > 1:
                    norm_dir = (dir_to_target[0] / dist, dir_to_target[1] / dist)
                    homing_force = 0.4
                    p['vel'][0] = (1 - homing_force) * p['vel'][0] + homing_force * norm_dir[0] * self.launch_power * 0.8
                    p['vel'][1] = (1 - homing_force) * p['vel'][1] + homing_force * norm_dir[1] * self.launch_power * 0.8
            else: # Normal projectiles affected by wind
                p['vel'][0] += self.wind_vector[0]
                p['vel'][1] += self.wind_vector[1]

            # Apply gravity
            p['vel'][1] += self.GRAVITY
            
            # Update position
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]

            # Proximity reward calculation
            dist_to_target = math.hypot(p['pos'][0] - self.target_pos[0], p['pos'][1] - self.target_pos[1])
            self.current_min_dist_to_target = min(self.current_min_dist_to_target, dist_to_target)
            
            # Check for target hit
            if dist_to_target < 25: # Target radius is 20
                self.target_hits += 1
                self.is_guided_missile_active = True
                reward += 10.0
                self._create_explosion(self.target_pos)
                projectiles_to_remove.append(i)
                continue

            # Check for out of bounds
            if not (0 < p['pos'][0] < self.SCREEN_WIDTH and p['pos'][1] < self.SCREEN_HEIGHT):
                projectiles_to_remove.append(i)
                # Proximity reward on miss
                improvement = self.last_launch_min_dist - self.current_min_dist_to_target
                reward += np.clip(improvement * 0.01, -2, 2)
                self.last_launch_min_dist = self.current_min_dist_to_target
                continue
        
        # Remove projectiles that hit or went OOB
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]
            
        return reward

    def _create_explosion(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = (math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                'pos': list(pos),
                'vel': list(vel),
                'life': self.np_random.integers(20, 40),
                'color': random.choice([(255, 200, 0), (255, 100, 0), (200, 50, 0)])
            })

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity on particles
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _check_termination(self):
        win = self.target_hits >= self.HITS_TO_WIN
        loss = self.remaining_ammo == 0 and not self.projectiles
        return win or loss

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "hits": self.target_hits,
            "ammo": self.remaining_ammo,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw static elements ---
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM, (0, self.SCREEN_HEIGHT - 20, self.SCREEN_WIDTH, 20))
        
        # --- Draw launcher ---
        barrel_length = 30
        angle_rad = math.radians(self.launch_angle)
        end_x = self.launcher_pos[0] + barrel_length * math.cos(angle_rad)
        end_y = self.launcher_pos[1] + barrel_length * math.sin(angle_rad)
        pygame.draw.line(self.screen, self.COLOR_LAUNCHER, self.launcher_pos, (end_x, end_y), 8)
        if pygame.gfxdraw:
            pygame.gfxdraw.aacircle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 10, self.COLOR_LAUNCHER)
            pygame.gfxdraw.filled_circle(self.screen, int(self.launcher_pos[0]), int(self.launcher_pos[1]), 10, self.COLOR_LAUNCHER)
        else:
            pygame.draw.circle(self.screen, self.COLOR_LAUNCHER, (int(self.launcher_pos[0]), int(self.launcher_pos[1])), 10)

        # --- Draw target ---
        if pygame.gfxdraw:
            pygame.gfxdraw.aacircle(self.screen, int(self.target_pos[0]), int(self.target_pos[1]), 20, self.COLOR_TARGET)
            pygame.gfxdraw.filled_circle(self.screen, int(self.target_pos[0]), int(self.target_pos[1]), 20, self.COLOR_TARGET)
            pygame.gfxdraw.aacircle(self.screen, int(self.target_pos[0]), int(self.target_pos[1]), 10, self.COLOR_BG)
            pygame.gfxdraw.filled_circle(self.screen, int(self.target_pos[0]), int(self.target_pos[1]), 10, self.COLOR_BG)
        else:
            pygame.draw.circle(self.screen, self.COLOR_TARGET, (int(self.target_pos[0]), int(self.target_pos[1])), 20)
            pygame.draw.circle(self.screen, self.COLOR_BG, (int(self.target_pos[0]), int(self.target_pos[1])), 10)


        # --- Draw trajectory prediction ---
        if not self.projectiles:
            self._render_prediction_line()

        # --- Draw projectiles and trails ---
        for p in self.projectiles:
            # Trail
            if len(p['trail']) > 1:
                trail_color = self.COLOR_GUIDED_PROJECTILE if p['type'] == 'guided' else self.COLOR_TRAIL
                try:
                    # aaline with alpha is not always available
                    for i in range(len(p['trail']) - 1):
                        alpha = int(255 * (i / len(p['trail'])))
                        pygame.draw.aaline(self.screen, trail_color + (alpha,), p['trail'][i], p['trail'][i+1])
                except (TypeError, ValueError):
                    pygame.draw.aalines(self.screen, trail_color, False, p['trail'])
            # Projectile
            color = self.COLOR_GUIDED_PROJECTILE if p['type'] == 'guided' else self.COLOR_PROJECTILE
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            if pygame.gfxdraw:
                pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], 6, color)
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 6, color)
            else:
                pygame.draw.circle(self.screen, color, pos_int, 6)
        
        # --- Draw particles ---
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            pos_int = (int(p['pos'][0]), int(p['pos'][1]))
            temp_surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, pos_int, int(p['life'] * 0.2))
            self.screen.blit(temp_surf, (0,0))


    def _render_prediction_line(self):
        sim_pos = list(self.launcher_pos)
        angle_rad = math.radians(self.launch_angle)
        sim_vel = [self.launch_power * math.cos(angle_rad), self.launch_power * math.sin(angle_rad)]
        points = []
        for _ in range(100):
            sim_vel[0] += self.wind_vector[0] * 0.5 # Show half wind effect for clarity
            sim_vel[1] += self.wind_vector[1] * 0.5
            sim_vel[1] += self.GRAVITY
            sim_pos[0] += sim_vel[0]
            sim_pos[1] += sim_vel[1]
            if _ % 3 == 0:
                points.append(tuple(map(int, sim_pos)))
            if not (0 < sim_pos[0] < self.SCREEN_WIDTH and 0 < sim_pos[1] < self.SCREEN_HEIGHT):
                break
        if len(points) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PREDICTION, False, points)

    def _render_ui(self):
        # --- Ammo Count ---
        ammo_text = self.font_small.render(f"AMMO: {self.remaining_ammo}/{self.MAX_AMMO}", True, self.COLOR_TEXT)
        self.screen.blit(ammo_text, (10, self.SCREEN_HEIGHT - 28))

        # --- Hits Count ---
        hits_text = self.font_small.render(f"HITS: {self.target_hits}/{self.HITS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(hits_text, (self.SCREEN_WIDTH - hits_text.get_width() - 10, 10))

        # --- Wind Indicator ---
        wind_strength = math.hypot(*self.wind_vector) * 1000
        wind_angle = math.atan2(self.wind_vector[1], self.wind_vector[0])
        wind_start = (self.SCREEN_WIDTH // 2, 20)
        wind_end = (wind_start[0] + wind_strength * math.cos(wind_angle), wind_start[1] + wind_strength * math.sin(wind_angle))
        pygame.draw.aaline(self.screen, self.COLOR_WIND, wind_start, wind_end, 2)
        if pygame.gfxdraw:
            pygame.gfxdraw.filled_circle(self.screen, int(wind_end[0]), int(wind_end[1]), 3, self.COLOR_WIND)
        else:
            pygame.draw.circle(self.screen, self.COLOR_WIND, (int(wind_end[0]), int(wind_end[1])), 3)
        wind_text = self.font_small.render("WIND", True, self.COLOR_WIND)
        self.screen.blit(wind_text, (wind_start[0] - wind_text.get_width()//2, wind_start[1] + 5))

        # --- Power Bar ---
        power_ratio = (self.launch_power - 3.0) / (15.0 - 3.0)
        bar_width = 100
        bar_height = 15
        bar_x = self.launcher_pos[0] - bar_width // 2
        bar_y = self.SCREEN_HEIGHT - 45
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_POWER_BAR_FILL, (bar_x, bar_y, bar_width * power_ratio, bar_height))

        # --- Guided Missile Indicator ---
        if self.is_guided_missile_active:
            guided_text = self.font_small.render("GUIDED MISSILE READY", True, self.COLOR_GUIDED_PROJECTILE)
            self.screen.blit(guided_text, (self.SCREEN_WIDTH // 2 - guided_text.get_width() // 2, self.SCREEN_HEIGHT - 70))
        
        # --- Game Over Text ---
        if self.game_over:
            if self.target_hits >= self.HITS_TO_WIN:
                msg = "YOU WIN!"
                color = self.COLOR_TARGET
            else:
                msg = "GAME OVER"
                color = self.COLOR_PROJECTILE
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    # This block allows you to play the game manually.
    # It will open a pygame window and let you control the agent.
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    display = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Artillery Game")
    
    running = True
    total_reward = 0
    
    action = [0, 0, 0] # [movement, space, shift]
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action[1] = 1
                if event.key == pygame.K_r: # Reset on 'r' key
                    print("Resetting environment.")
                    obs, info = env.reset()
                    total_reward = 0
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    action[1] = 0

        keys = pygame.key.get_pressed()
        action[0] = 0 # Default no movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            print("Press 'R' to reset.")

        env.clock.tick(env.metadata["render_fps"])
        
    env.close()