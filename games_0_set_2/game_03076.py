
# Generated: 2025-08-27T22:18:18.840021
# Source Brief: brief_03076.md
# Brief Index: 3076

        
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
    """
    A minimalist top-down space shooter where the player destroys waves of descending aliens.
    This environment prioritizes visual quality and satisfying "game feel" with features like
    particle effects, anti-aliasing, and smooth animations, designed for a 30 FPS frame rate.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Short, user-facing control string
    user_guide = (
        "Controls: Use ← and → arrow keys to move. Hold the space bar to fire."
    )

    # Short, user-facing description of the game
    game_description = (
        "A retro arcade shooter. Survive 20 waves of descending aliens to win."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 10000
        self.MAX_WAVES = 20
        
        # Colors (Bright for interactive, Dark for background)
        self.COLOR_BG = (10, 15, 25)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_P_PROJ = (180, 255, 180)
        self.COLOR_E_PROJ = (255, 100, 100)
        self.COLOR_ALIEN_A = (80, 220, 120)  # Green
        self.COLOR_ALIEN_B = (100, 150, 255) # Blue
        self.COLOR_ALIEN_C = (255, 220, 80)  # Yellow
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_TEXT_SHADOW = (50, 50, 50)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup (Headless) ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 52)
        self.font_medium = pygame.font.Font(None, 28)
        
        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave = 0
        self.player_lives = 0
        self.player_pos = [0, 0]
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.aliens = []
        self.particles = []
        self.player_shoot_cooldown = 0
        self.player_invincibility_timer = 0
        self.wave_transition_timer = 0
        self.np_random = None

        # Initialize state variables
        self.reset()
        
        # Validate implementation after initialization
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.wave = 1
        self.player_lives = 3
        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 40]
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.aliens.clear()
        self.particles.clear()
        self.player_shoot_cooldown = 0
        self.player_invincibility_timer = 0
        self.wave_transition_timer = 120 # Start with a "Wave 1" message display

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.01  # Small penalty per frame to encourage efficiency
        terminated = False

        if not self.game_over:
            self.steps += 1
            
            # If in a wave transition, just count down the timer
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
                if self.wave_transition_timer == 0 and self.wave > 1 and not self.game_won:
                    self._spawn_wave()
            else: # Normal gameplay logic
                self._handle_input(action)
                reward += self._update_game_state()
                reward += self._handle_collisions()

                # Check for wave clear condition
                if not self.aliens:
                    reward += 5.0
                    self.score += 50 * self.wave
                    self.wave += 1
                    if self.wave > self.MAX_WAVES:
                        self.game_won = True
                        self.game_over = True
                        reward += 100.0  # Big reward for winning
                    else:
                        self.wave_transition_timer = 90  # 3-second pause for "WAVE CLEARED"

        # Check for termination conditions
        if self.player_lives <= 0 and not self.game_over:
            reward += -100.0  # Big penalty for losing
            self.game_over = True
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated is always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Horizontal Movement
        if movement == 3:  # Left
            self.player_pos[0] -= 8
        elif movement == 4:  # Right
            self.player_pos[0] += 8
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.SCREEN_WIDTH - 20)

        # Firing Projectile
        if space_held and self.player_shoot_cooldown <= 0:
            # sfx: player_shoot.wav
            proj_rect = pygame.Rect(self.player_pos[0] - 2, self.player_pos[1] - 20, 4, 15)
            self.player_projectiles.append(proj_rect)
            self.player_shoot_cooldown = 8  # Cooldown in frames

    def _update_game_state(self):
        life_lost_penalty = 0.0

        # Update player state
        if self.player_shoot_cooldown > 0: self.player_shoot_cooldown -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1

        # Update player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= 12
            if proj.bottom < 0: self.player_projectiles.remove(proj)
        
        # Update enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj.y += 6
            if proj.top > self.SCREEN_HEIGHT: self.enemy_projectiles.remove(proj)

        # Update aliens
        speed_mod = 1.0 + (self.wave - 1) * 0.05
        fire_rate = 0.01 + (self.wave - 1) * 0.001

        for alien in self.aliens[:]:
            # Movement patterns
            if alien['type'] == 'A': # Linear
                alien['rect'].y += 1.5 * speed_mod
            elif alien['type'] == 'B': # Sinusoidal
                alien['pattern_phase'] += 0.05
                alien['rect'].x = alien['start_x'] + math.sin(alien['pattern_phase']) * 60
                alien['rect'].y += 1.0 * speed_mod
            elif alien['type'] == 'C': # Fast Linear
                alien['rect'].y += 2.0 * speed_mod

            # Firing logic
            if self.np_random.random() < fire_rate:
                # sfx: enemy_shoot.wav
                proj_rect = pygame.Rect(alien['rect'].centerx - 2, alien['rect'].bottom, 4, 10)
                self.enemy_projectiles.append(proj_rect)
            
            # Alien reaches bottom of screen
            if alien['rect'].top > self.SCREEN_HEIGHT:
                self.aliens.remove(alien)
                if self.player_invincibility_timer <= 0:
                    self.player_lives -= 1
                    # sfx: player_hit.wav
                    self.player_invincibility_timer = 120 # 4s invincibility
                    life_lost_penalty -= 25.0 # Penalty for letting alien pass

        # Update particles for explosions
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0: self.particles.remove(p)
        
        return life_lost_penalty

    def _handle_collisions(self):
        reward = 0.0

        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.colliderect(alien['rect']):
                    # sfx: alien_destroyed.wav
                    self._create_explosion(alien['rect'].center, alien['color'])
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    self.score += 10
                    reward += 1.0  # Reward for destroying an alien
                    break

        # Enemy projectiles vs player
        if self.player_invincibility_timer <= 0:
            player_hitbox = pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 8, 24, 16)
            for proj in self.enemy_projectiles[:]:
                if player_hitbox.colliderect(proj):
                    # sfx: player_hit.wav
                    self.enemy_projectiles.remove(proj)
                    self.player_lives -= 1
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                    if self.player_lives > 0:
                        self.player_invincibility_timer = 120 # 4s invincibility
                    break
        
        return reward

    def _spawn_wave(self):
        self.aliens.clear()
        num_aliens = min(8 + self.wave * 2, 40)
        rows = (num_aliens // 8) + 1
        cols = 8
        
        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            
            x = self.SCREEN_WIDTH * (col + 1) / (cols + 2)
            y = 60 + row * 40
            
            rand_val = self.np_random.random()
            if rand_val < 0.5:
                atype, acolor = 'A', self.COLOR_ALIEN_A
            elif rand_val < 0.85:
                atype, acolor = 'B', self.COLOR_ALIEN_B
            else:
                atype, acolor = 'C', self.COLOR_ALIEN_C
                
            self.aliens.append({
                'rect': pygame.Rect(x - 12, y - 12, 24, 24),
                'type': atype,
                'color': acolor,
                'start_x': x,
                'pattern_phase': self.np_random.random() * 2 * math.pi
            })

    def _create_explosion(self, pos, color):
        num_particles = 25
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.integers(2, 5)
            })
    
    def _render_text(self, text, font, x, y, center=False):
        text_surf = font.render(text, True, self.COLOR_TEXT)
        shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surf, text_rect)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, alien['color'], alien['rect'], border_radius=3)
            pygame.gfxdraw.rectangle(self.screen, alien['rect'], tuple(c*0.7 for c in alien['color']))

        # Render projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_P_PROJ, proj, border_radius=2)
        for proj in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_E_PROJ, proj, border_radius=2)

        # Render player ship
        is_invincible = self.player_invincibility_timer > 0
        is_visible = not (is_invincible and (self.steps // 3) % 2 == 0)
        if is_visible and self.player_lives > 0:
            p = [int(self.player_pos[0]), int(self.player_pos[1])]
            points = [(p[0], p[1]-12), (p[0]-12, p[1]+8), (p[0]+12, p[1]+8)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Render particles
        for p in self.particles:
            alpha = max(0, int(255 * (p['lifespan'] / 30)))
            size = max(0, int(p['size'] * (p['lifespan'] / 30)))
            if size > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, (*p['color'], alpha))

    def _render_ui(self):
        # Score and Wave
        self._render_text(f"SCORE: {self.score}", self.font_medium, 15, 10)
        wave_str = f"WAVE: {self.wave}/{self.MAX_WAVES}"
        wave_w = self.font_medium.size(wave_str)[0]
        self._render_text(wave_str, self.font_medium, self.SCREEN_WIDTH - wave_w - 15, 10)

        # Player Lives
        for i in range(self.player_lives):
            p = [40 + i * 30, self.SCREEN_HEIGHT - 20]
            points = [(p[0], p[1]-8), (p[0]-8, p[1]+5), (p[0]+8, p[1]+5)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.game_won else "GAME OVER"
            self._render_text(msg, self.font_large, self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2, center=True)
        
        # Wave transition message
        elif self.wave_transition_timer > 30:
            msg = f"WAVE {self.wave}"
            if len(self.aliens) == 0 and self.wave > 1:
                msg = "WAVE CLEARED"
            self._render_text(msg, self.font_large, self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "lives": self.player_lives,
        }
        
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game directly
    # Requires pygame to be installed with display support
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Space Shooter")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 0 # Unused in this game
        
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Info: {info}")
    print(f"Total Reward: {total_reward:.2f}")
    
    env.close()