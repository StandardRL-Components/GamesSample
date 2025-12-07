import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:38:25.693363
# Source Brief: brief_00538.md
# Brief Index: 538
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Launch a gravity-defying orb to smash all the targets. "
        "Flip gravity on the fly and use slow-motion to line up the perfect shot."
    )
    user_guide = (
        "Use arrow keys to aim. Press space to launch the orb. "
        "Once launched, press space to flip gravity. Hold shift to activate slow-motion."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_GRID = (20, 15, 50)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_ENEMY = (128, 0, 128)
    COLOR_ENEMY_GLOW = (70, 0, 70)
    COLOR_TEXT = (220, 220, 220)
    COLOR_GRAVITY_NORMAL = (0, 255, 0)
    COLOR_GRAVITY_FLIPPED = (255, 0, 0)
    COLOR_SLOWMO_BAR = (0, 100, 255)
    COLOR_SLOWMO_BAR_BG = (50, 50, 50)

    # Physics & Gameplay
    GRAVITY_STRENGTH = 0.35
    PLAYER_LAUNCH_SPEED = 8.0
    PLAYER_RADIUS = 12
    ENEMY_RADIUS = 15
    WALL_DAMPING = 0.85
    MAX_STEPS = 2000
    SLOWMO_DRAIN_RATE = 1.0
    SLOWMO_RECHARGE_RATE = 0.3
    SLOWMO_FACTOR = 0.25
    PARTICLE_LIFESPAN = 25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.level = 0 # Will be set to 1 in the first reset
        self.game_over = False
        self.player_pos = None
        self.player_vel = None
        self.enemies = []
        self.particles = []
        self.gravity = None
        self.orb_launched = False
        self.aim_angle = 0
        self.slowmo_charge = 100.0
        self.is_slowmo_active = False
        self.last_space_state = 0
        self.level_cleared_timer = 0
        
        # self.reset() # reset is called by the environment runner
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.game_over or self.level == 0:
            self.score = 0
            self.level = 1
        else: # Level cleared
            self.level += 1

        self.steps = 0
        self.game_over = False
        self.level_cleared_timer = 0
        
        self.orb_launched = False
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40], dtype=np.float64)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float64)
        self.aim_angle = -math.pi / 2  # Start aiming up
        
        self.gravity = np.array([0.0, self.GRAVITY_STRENGTH], dtype=np.float64)
        self.slowmo_charge = 100.0
        self.is_slowmo_active = False
        self.last_space_state = 0

        self.enemies.clear()
        num_enemies = 3 + (self.level -1) // 2
        enemy_speed = 1.0 + (self.level -1) * 0.05
        for _ in range(num_enemies):
            self._spawn_enemy(enemy_speed)

        self.particles.clear()
        
        return self._get_observation(), self._get_info()

    def _spawn_enemy(self, speed):
        padding = 50
        pos = np.array([
            random.uniform(padding, self.SCREEN_WIDTH - padding),
            random.uniform(padding, self.SCREEN_HEIGHT / 2)
        ], dtype=np.float64)
        self.enemies.append({
            "pos": pos,
            "radius": self.ENEMY_RADIUS,
            "speed": speed,
            "phase_offset": random.uniform(0, 2 * math.pi)
        })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        if self.level_cleared_timer > 0:
            self.level_cleared_timer -= 1
            if self.level_cleared_timer == 0:
                # The episode ends here, and the environment will be reset externally
                # We give a large reward for clearing the level
                return self._get_observation(), 100.0, True, False, self._get_info()
            # While waiting, do nothing and return a non-terminal state
            return self._get_observation(), 0, False, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_action, shift_held = action[0], action[1], action[2] == 1
        space_pressed = space_action == 1 and self.last_space_state == 0
        self.last_space_state = space_action

        # Handle Input
        self._handle_input(movement, space_pressed, shift_held)

        # Determine time scale
        time_delta = 1.0
        if self.is_slowmo_active:
            self.slowmo_charge = max(0, self.slowmo_charge - self.SLOWMO_DRAIN_RATE)
            if self.slowmo_charge == 0:
                self.is_slowmo_active = False
            time_delta = self.SLOWMO_FACTOR
        else:
            self.slowmo_charge = min(100, self.slowmo_charge + self.SLOWMO_RECHARGE_RATE)

        # Update Game Logic
        self._update_physics(time_delta)
        self._update_enemies(time_delta)
        self._update_particles()
        
        # Handle Collisions and calculate rewards from them
        collision_reward, destroyed_enemies, is_game_over = self._handle_collisions()
        reward += collision_reward
        self.game_over = is_game_over

        if not self.orb_launched:
            reward -= 0.01

        # Check Termination Conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.game_over:
            reward = -100.0

        if not self.enemies and self.orb_launched and not self.game_over:
            self.level_cleared_timer = self.FPS * 2 # 2 second pause
            # The final +100 reward will be given when the timer runs out and the episode truly ends
            reward += 10.0 # Intermediate reward for clearing
            # The episode is not over yet, just waiting for the timer.
            # The next step will enter the level_cleared_timer logic.
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_pressed, shift_held):
        # Slow-mo
        if shift_held and self.slowmo_charge > 0:
            self.is_slowmo_active = True
        else:
            self.is_slowmo_active = False

        # Aiming (only if orb not launched)
        if not self.orb_launched:
            aim_speed = 0.1
            if movement == 1: self.aim_angle -= aim_speed # Up
            if movement == 2: self.aim_angle += aim_speed # Down
            if movement == 3: self.aim_angle -= aim_speed # Left
            if movement == 4: self.aim_angle += aim_speed # Right
        
        # Gravity Flip / Launch
        if space_pressed:
            # First press launches the orb
            if not self.orb_launched:
                self.orb_launched = True
                self.player_vel[0] = math.cos(self.aim_angle) * self.PLAYER_LAUNCH_SPEED
                self.player_vel[1] = math.sin(self.aim_angle) * self.PLAYER_LAUNCH_SPEED
                # sfx: launch_orb
            # All presses flip gravity
            self.gravity *= -1
            # sfx: gravity_flip

    def _update_physics(self, time_delta):
        if not self.orb_launched:
            return

        # Apply gravity
        self.player_vel += self.gravity * time_delta
        
        # Update position
        self.player_pos += self.player_vel * time_delta

        # Wall collisions
        if self.player_pos[0] < self.PLAYER_RADIUS:
            self.player_pos[0] = self.PLAYER_RADIUS
            self.player_vel[0] *= -self.WALL_DAMPING
            self._create_particles(self.player_pos, 5) # sfx: wall_bounce
        elif self.player_pos[0] > self.SCREEN_WIDTH - self.PLAYER_RADIUS:
            self.player_pos[0] = self.SCREEN_WIDTH - self.PLAYER_RADIUS
            self.player_vel[0] *= -self.WALL_DAMPING
            self._create_particles(self.player_pos, 5) # sfx: wall_bounce

        if self.player_pos[1] < self.PLAYER_RADIUS:
            self.player_pos[1] = self.PLAYER_RADIUS
            self.player_vel[1] *= -self.WALL_DAMPING
            self._create_particles(self.player_pos, 5) # sfx: wall_bounce
        elif self.player_pos[1] > self.SCREEN_HEIGHT - self.PLAYER_RADIUS:
            self.player_pos[1] = self.SCREEN_HEIGHT - self.PLAYER_RADIUS
            self.player_vel[1] *= -self.WALL_DAMPING
            self._create_particles(self.player_pos, 5) # sfx: wall_bounce

    def _update_enemies(self, time_delta):
        for enemy in self.enemies:
            # Sinusoidal homing behavior
            direction_to_player = self.player_pos - enemy["pos"]
            dist = np.linalg.norm(direction_to_player)
            if dist > 1.0:
                direction_to_player /= dist
            
            perp_vector = np.array([-direction_to_player[1], direction_to_player[0]])
            sine_wave = math.sin(self.steps * 0.1 + enemy["phase_offset"]) * 0.6
            
            final_direction = direction_to_player + perp_vector * sine_wave
            final_dist = np.linalg.norm(final_direction)
            if final_dist > 1.0:
                 final_direction /= final_dist

            enemy["pos"] += final_direction * enemy["speed"] * time_delta

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0.0
        destroyed_enemies = 0
        game_over = False

        # Player Orb vs Enemies
        enemies_to_remove = []
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy["pos"])
            if dist < self.PLAYER_RADIUS + enemy["radius"]:
                if self.orb_launched:
                    # Player destroys enemy
                    enemies_to_remove.append(enemy)
                    self.score += 100
                    reward += 1.1 # +1 for destroy, +0.1 for hit
                    destroyed_enemies += 1
                    self._create_particles(enemy["pos"], 20, self.COLOR_ENEMY) # sfx: enemy_destroy
                else:
                    # Enemy consumes player orb (Game Over)
                    game_over = True
                    self._create_particles(self.player_pos, 50, self.COLOR_PLAYER) # sfx: player_death
                    break
        
        if enemies_to_remove:
            # Using object IDs for removal is safe when objects contain numpy arrays,
            # as it avoids element-wise comparison that leads to ValueError.
            ids_to_remove = {id(e) for e in enemies_to_remove}
            self.enemies = [e for e in self.enemies if id(e) not in ids_to_remove]
        
        return reward, destroyed_enemies, game_over

    def _create_particles(self, position, count, color=None):
        if color is None:
            color = self.COLOR_PLAYER
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                "pos": position.copy(),
                "vel": vel,
                "life": self.PARTICLE_LIFESPAN,
                "color": color
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        if self.is_slowmo_active:
            self._render_slowmo_effect()

        self._render_particles()
        self._render_enemies()
        self._render_player()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT), 1)
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y), 1)
        pygame.draw.rect(self.screen, self.COLOR_GRID, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 2)

    def _render_slowmo_effect(self):
        vignette = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        radial_alpha = 100 - (self.slowmo_charge)
        pygame.draw.rect(vignette, (*self.COLOR_SLOWMO_BAR, radial_alpha), (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 20)
        self.screen.blit(vignette, (0, 0))

    def _render_player(self):
        pos_int = self.player_pos.astype(int)
        
        # Trail
        if self.orb_launched and np.linalg.norm(self.player_vel) > 1:
            trail_length = 5
            for i in range(trail_length):
                p = i / trail_length
                prev_pos = self.player_pos - self.player_vel * p * 0.5
                alpha = 150 * (1 - p)
                radius = self.PLAYER_RADIUS * (1 - p * 0.5)
                self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, prev_pos.astype(int), int(radius), glow_factor=1.5, glow_alpha=int(alpha/3))

        # Main Orb
        self._draw_glowing_circle(self.screen, self.COLOR_PLAYER, pos_int, self.PLAYER_RADIUS, glow_factor=2.0)

        # Aimer
        if not self.orb_launched:
            aim_len = 50
            end_pos = self.player_pos + np.array([math.cos(self.aim_angle), math.sin(self.aim_angle)]) * aim_len
            pygame.draw.aaline(self.screen, self.COLOR_PLAYER, pos_int, end_pos.astype(int), 2)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = enemy["pos"].astype(int)
            # Amorphous look with jittering circles
            for i in range(3):
                offset_angle = (self.steps * 0.2 + i * 2.1)
                offset = np.array([math.cos(offset_angle), math.sin(offset_angle)]) * 3
                self._draw_glowing_circle(self.screen, self.COLOR_ENEMY, (pos_int + offset).astype(int), enemy["radius"], glow_factor=1.5, glow_alpha=50)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / self.PARTICLE_LIFESPAN))))
            color = (*p['color'], alpha)
            size = int(max(1, 5 * (p['life'] / self.PARTICLE_LIFESPAN)))
            
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, p['pos'] - size)
            
    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Level
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        # Gravity Indicator
        grav_color = self.COLOR_GRAVITY_NORMAL if self.gravity[1] > 0 else self.COLOR_GRAVITY_FLIPPED
        arrow_points = []
        if self.gravity[1] > 0: # Down
            arrow_points = [(self.SCREEN_WIDTH // 2, 35), (self.SCREEN_WIDTH // 2 - 10, 25), (self.SCREEN_WIDTH // 2 + 10, 25)]
        else: # Up
            arrow_points = [(self.SCREEN_WIDTH // 2, 25), (self.SCREEN_WIDTH // 2 - 10, 35), (self.SCREEN_WIDTH // 2 + 10, 35)]
        pygame.draw.polygon(self.screen, grav_color, arrow_points)
        pygame.gfxdraw.aapolygon(self.screen, arrow_points, grav_color)

        # Slow-mo Bar
        bar_width, bar_height = 200, 15
        bar_x = (self.SCREEN_WIDTH - bar_width) // 2
        bar_y = self.SCREEN_HEIGHT - bar_height - 10
        fill_width = (self.slowmo_charge / 100) * bar_width
        pygame.draw.rect(self.screen, self.COLOR_SLOWMO_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=4)
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_SLOWMO_BAR, (bar_x, bar_y, fill_width, bar_height), border_radius=4)

        # Game Over / Level Clear Text
        if self.game_over:
            text = self.font_main.render("GAME OVER", True, self.COLOR_GRAVITY_FLIPPED)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)
        elif self.level_cleared_timer > 0:
            text = self.font_main.render("LEVEL CLEARED", True, self.COLOR_GRAVITY_NORMAL)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(text, text_rect)

    def _draw_glowing_circle(self, surface, color, center, radius, glow_factor=2.0, glow_alpha=70):
        # Draw the glow
        glow_radius = int(radius * glow_factor)
        glow_color = (*color, glow_alpha)
        
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (center[0] - glow_radius, center[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
        
        # Draw the main circle
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "orb_launched": self.orb_launched,
            "enemies_left": len(self.enemies)
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires a graphical display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Action defaults
        movement = 0 # no-op
        space = 0
        shift = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
        except Exception:
            display_surf = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        
        display_surf.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Level: {info['level']}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        env.clock.tick(GameEnv.FPS)

    env.close()