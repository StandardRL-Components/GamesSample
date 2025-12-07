import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:07:24.290830
# Source Brief: brief_00896.md
# Brief Index: 896
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player morphs between a spider and a beetle
    to dodge waves of projectiles and reach an exit.

    **Visuals:** Minimalist, geometric, neon-colored on a dark background.
    **Gameplay:** Dodge projectiles by moving and changing form. The Spider is fast
    but fragile. The Beetle is slow but can withstand one hit.
    **Objective:** Reach the yellow exit zone within 60 seconds without being destroyed.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Morph between a fast spider and a durable beetle to dodge waves of projectiles and reach the exit."
    user_guide = "Controls: Use arrow keys (↑↓←→) to move. Press space to morph between spider and beetle forms."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 60 * FPS  # 60 seconds

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_GLOW = (0, 255, 128, 50)
    COLOR_BEETLE_SHIELD_BROKEN = (255, 200, 0)
    COLOR_PROJ_RED = (255, 50, 50)
    COLOR_PROJ_BLUE = (50, 150, 255)
    COLOR_PROJ_GLOW_RED = (255, 50, 50, 60)
    COLOR_PROJ_GLOW_BLUE = (50, 150, 255, 60)
    COLOR_EXIT = (255, 220, 0)
    COLOR_EXIT_GLOW = (255, 220, 0, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_PROGRESS_BAR = (0, 128, 255)
    COLOR_PROGRESS_BAR_BG = (40, 50, 70)

    # Player settings
    PLAYER_SPIDER_SPEED = 4.0
    PLAYER_BEETLE_SPEED = 2.0
    PLAYER_SPIDER_HITBOX = 12
    PLAYER_BEETLE_HITBOX = 10

    # Projectile settings
    PROJECTILE_BASE_SPEED = 1.5
    PROJECTILE_SPEED_INCREMENT = 0.05
    PROJECTILE_RADIUS = 6

    # Game flow
    WAVE_DURATION_FRAMES = 4 * FPS
    PATTERN_SWITCH_FRAMES = 10 * FPS
    PATTERN_COOLDOWN_FRAMES = 2 * FPS
    TOTAL_WAVES = 15

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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        # --- State Variables ---
        # These are initialized here but properly set in reset()
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = 0.0

        self.player_pos = pygame.math.Vector2(0, 0)
        self.player_form = 0  # 0: spider, 1: beetle
        self.beetle_shield_active = True
        self.prev_space_held = False

        self.exit_rect = pygame.Rect(0, 0, 0, 0)
        
        self.projectiles = []
        self.particles = []
        self.projectile_speed = 0.0
        
        self.wave_number = 0
        self.wave_timer = 0
        self.pattern_timer = 0
        self.current_pattern = 0 # 0: horizontal, 1: vertical
        self.is_in_cooldown = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer = self.MAX_STEPS

        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50)
        self.player_form = 0  # Start as spider
        self.beetle_shield_active = True
        self.prev_space_held = False

        self.exit_rect = pygame.Rect(self.SCREEN_WIDTH / 2 - 20, 10, 40, 40)
        
        self.projectiles.clear()
        self.particles.clear()
        self.projectile_speed = self.PROJECTILE_BASE_SPEED
        
        self.wave_number = 0
        self.wave_timer = self.WAVE_DURATION_FRAMES
        self.pattern_timer = self.PATTERN_SWITCH_FRAMES
        self.current_pattern = self.np_random.integers(0, 2)
        self.is_in_cooldown = True # Start with cooldown

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0.1 # Survival reward

        self._handle_input(action)
        self._update_player()
        self._update_projectiles()
        self._update_particles()
        
        # Check game state changes
        player_hit = self._check_collisions()
        reached_exit = self.exit_rect.collidepoint(self.player_pos.x, self.player_pos.y)
        timeout = self.timer <= 0

        terminated = False
        truncated = False # In this env, truncated is always false

        if player_hit:
            reward = -100.0
            self.score -= 100
            self.game_over = True
            terminated = True
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 50)
        elif reached_exit:
            reward = 100.0
            self.score += 100
            self.game_over = True
            terminated = True
        elif timeout:
            self.game_over = True
            terminated = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held_raw, _ = action
        space_held = space_held_raw == 1

        # --- Movement ---
        move_vector = pygame.math.Vector2(0, 0)
        if movement == 1: move_vector.y = -1  # Up
        elif movement == 2: move_vector.y = 1   # Down
        elif movement == 3: move_vector.x = -1  # Left
        elif movement == 4: move_vector.x = 1   # Right
        
        speed = self.PLAYER_SPIDER_SPEED if self.player_form == 0 else self.PLAYER_BEETLE_SPEED
        if move_vector.length() > 0:
            move_vector.normalize_ip()
            self.player_pos += move_vector * speed

        # --- Form Switching (Morph) ---
        if space_held and not self.prev_space_held:
            self.player_form = 1 - self.player_form # Toggle 0 and 1
            if self.player_form == 1: # Switched to beetle
                self.beetle_shield_active = True # Reset shield
            self._create_explosion(self.player_pos, self.COLOR_UI_TEXT, 15, 2.0)
        self.prev_space_held = space_held

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.SCREEN_WIDTH)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.SCREEN_HEIGHT)

    def _update_projectiles(self):
        # --- Spawning Logic ---
        self.wave_timer -= 1
        self.pattern_timer -= 1

        if self.pattern_timer <= 0:
            self.current_pattern = 1 - self.current_pattern
            self.pattern_timer = self.PATTERN_SWITCH_FRAMES
            self.is_in_cooldown = True
            self.wave_timer = self.PATTERN_COOLDOWN_FRAMES # Set cooldown timer
        
        if self.is_in_cooldown and self.wave_timer <= 0:
            self.is_in_cooldown = False
            self.wave_timer = self.WAVE_DURATION_FRAMES # Reset wave timer
        
        if not self.is_in_cooldown and self.wave_timer <= 0 and self.wave_number < self.TOTAL_WAVES:
            self.wave_number += 1
            self.wave_timer = self.WAVE_DURATION_FRAMES
            self.projectile_speed += self.PROJECTILE_SPEED_INCREMENT
            self._spawn_wave()

        # --- Movement and Removal ---
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            if not self.screen.get_rect().collidepoint(proj['pos'].x, proj['pos'].y):
                self.projectiles.remove(proj)
                self.score += 1 # Reward for projectile going off-screen (dodged)

    def _spawn_wave(self):
        if self.current_pattern == 0: # Horizontal
            num_proj = self.np_random.integers(4, 7)
            for i in range(num_proj):
                y = self.np_random.uniform(50, self.SCREEN_HEIGHT - 50)
                side = self.np_random.choice([-1, 1])
                x = -20 if side == 1 else self.SCREEN_WIDTH + 20
                vel = pygame.math.Vector2(side * self.projectile_speed, self.np_random.uniform(-0.2, 0.2))
                self.projectiles.append({'pos': pygame.math.Vector2(x, y), 'vel': vel, 'type': 0})
        else: # Vertical
            num_proj = self.np_random.integers(6, 10)
            for i in range(num_proj):
                x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
                y = -20
                vel = pygame.math.Vector2(self.np_random.uniform(-0.2, 0.2), self.projectile_speed)
                self.projectiles.append({'pos': pygame.math.Vector2(x, y), 'vel': vel, 'type': 1})

    def _check_collisions(self):
        hitbox = self.PLAYER_SPIDER_HITBOX if self.player_form == 0 else self.PLAYER_BEETLE_HITBOX
        for proj in self.projectiles:
            if self.player_pos.distance_to(proj['pos']) < hitbox + self.PROJECTILE_RADIUS:
                if self.player_form == 0: # Spider
                    return True
                elif self.player_form == 1: # Beetle
                    if self.beetle_shield_active:
                        self.beetle_shield_active = False
                        self.projectiles.remove(proj)
                        self._create_explosion(self.player_pos, self.COLOR_BEETLE_SHIELD_BROKEN, 20)
                        return False # Not a fatal hit
                    else:
                        return True # Shield was already broken
        return False

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1.0, 3.0) * speed_mult
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'life': life, 'max_life': life, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2))

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "timer": self.timer, "wave": self.wave_number}
    
    def _render_game(self):
        self._render_exit()
        self._render_projectiles()
        self._render_particles()
        if not self.game_over:
            self._render_player()

    def _render_exit(self):
        center = self.exit_rect.center
        for i in range(10, 0, -1):
            alpha = self.COLOR_EXIT_GLOW[3] * (1 - i/10)
            pygame.gfxdraw.rectangle(self.screen, self.exit_rect.inflate(i*2, i*2), (*self.COLOR_EXIT, int(alpha)))
        pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect, 2)

    def _render_player(self):
        pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        
        # Glow
        for i in range(15, 0, -2):
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], i, self.COLOR_PLAYER_GLOW)

        if self.player_form == 0: # Spider
            # Body
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, pos_int, 5)
            # Legs
            for i in range(8):
                angle = (i * 45 + self.steps * 5) * (math.pi / 180)
                start = self.player_pos + pygame.math.Vector2(math.cos(angle), math.sin(angle)) * 4
                end = self.player_pos + pygame.math.Vector2(math.cos(angle * 1.1), math.sin(angle * 1.1)) * 14
                pygame.draw.aaline(self.screen, self.COLOR_PLAYER, start, end)
        else: # Beetle
            color = self.COLOR_PLAYER if self.beetle_shield_active else self.COLOR_BEETLE_SHIELD_BROKEN
            rect = pygame.Rect(0, 0, 20, 24)
            rect.center = pos_int
            pygame.draw.ellipse(self.screen, color, rect)
            pygame.draw.ellipse(self.screen, self.COLOR_BG, rect, 2)
            pygame.draw.line(self.screen, self.COLOR_BG, rect.midtop, rect.midbottom, 2)

    def _render_projectiles(self):
        for proj in self.projectiles:
            pos_int = (int(proj['pos'].x), int(proj['pos'].y))
            color = self.COLOR_PROJ_RED if proj['type'] == 0 else self.COLOR_PROJ_BLUE
            glow_color = self.COLOR_PROJ_GLOW_RED if proj['type'] == 0 else self.COLOR_PROJ_GLOW_BLUE
            
            for i in range(self.PROJECTILE_RADIUS * 2, self.PROJECTILE_RADIUS, -1):
                 pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], i, glow_color)
            
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PROJECTILE_RADIUS, color)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            radius = int(life_ratio * 5)
            if radius > 0:
                pos_int = (int(p['pos'].x), int(p['pos'].y))
                alpha_color = (*p['color'][:3], int(life_ratio * 255))
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], radius, alpha_color)

    def _render_ui(self):
        # --- Score and Timer ---
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        time_left = max(0, self.timer / self.FPS)
        timer_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # --- Wave Progress Bar ---
        bar_width = 200
        bar_height = 10
        bar_x = self.SCREEN_WIDTH / 2 - bar_width / 2
        bar_y = 15
        
        # Cooldown or active wave progress
        if self.is_in_cooldown:
            progress = 1.0 - (self.wave_timer / self.PATTERN_COOLDOWN_FRAMES)
            color = self.COLOR_UI_TEXT
        else:
            progress = 1.0 - (self.wave_timer / self.WAVE_DURATION_FRAMES)
            color = self.COLOR_PROGRESS_BAR
        
        fill_width = int(bar_width * progress)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))

        # --- Game Over Text ---
        if self.game_over:
            if self.exit_rect.collidepoint(self.player_pos.x, self.player_pos.y):
                msg = "SUCCESS"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_PROJ_RED
            
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # It will not run when the environment is used by the test suite
    real_render = "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy"
    if real_render:
        pygame.display.init()
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Morph Dodger")
    else:
        # The environment will render to a surface anyway, but we won't see it
        screen = None

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    clock = pygame.time.Clock()

    total_reward = 0
    
    while running:
        # --- Human Input Processing ---
        action = [0, 0, 0] # Default no-op
        if real_render:
            keys = pygame.key.get_pressed()
            movement = 0 # No-op
            if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
            elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        if real_render and screen:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
            if real_render:
                # Wait for 'R' to reset or quit
                wait_for_reset = True
                while wait_for_reset:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            wait_for_reset = False
                            running = False
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                            obs, info = env.reset()
                            total_reward = 0
                            wait_for_reset = False
            else: # In headless mode, just stop after one episode
                running = False
        
        clock.tick(GameEnv.FPS)

    env.close()