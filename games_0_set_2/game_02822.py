
# Generated: 2025-08-27T21:32:19.788895
# Source Brief: brief_02822.md
# Brief Index: 2822

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade shooter where the player must destroy waves of descending aliens
    while dodging their projectiles. The game features three stages of increasing
    difficulty, a temporary shield mechanic, and a scoring system that rewards
    skilled play.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Arrow keys to move. Hold shift for a temporary shield. Press space to fire."
    )
    game_description = (
        "A top-down shooter where you must destroy waves of descending aliens while dodging their projectiles."
    )

    # Frame advance behavior
    auto_advance = True

    # --- Constants ---
    # Screen
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_SHIELD = (64, 160, 255, 100)
    COLOR_PLAYER_PROJECTILE = (0, 220, 255)
    COLOR_ALIEN = (255, 60, 60)
    COLOR_ALIEN_PROJECTILE = (255, 165, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_STARS = [(255, 255, 255), (200, 200, 200), (150, 150, 150)]
    # Game parameters
    PLAYER_SPEED = 6
    PLAYER_LIVES = 3
    PLAYER_SHOOT_COOLDOWN = 6  # frames
    PLAYER_SHIELD_DURATION = 15 # frames
    PLAYER_SHIELD_COOLDOWN = 120 # frames
    PLAYER_INVINCIBILITY_DURATION = 60 # frames
    PROJECTILE_SPEED = 10
    ALIENS_PER_STAGE = 50
    ALIENS_PER_WAVE = 10
    TOTAL_STAGES = 3
    STAGE_TIME_LIMIT_SECONDS = 60
    FPS = 30

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
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.stars = []
        self._init_stars()

        # Initialize state variables (will be properly set in reset)
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_projectiles = []
        self.player_shoot_timer = 0
        self.player_shield_active = False
        self.player_shield_timer = 0
        self.player_shield_cooldown_timer = 0
        self.player_invincibility_timer = 0
        
        self.aliens = []
        self.alien_projectiles = []
        self.explosions = []

        self.score = 0
        self.steps = 0
        self.stage = 1
        self.aliens_destroyed_total = 0
        self.aliens_destroyed_this_stage = 0
        self.stage_timer = 0
        self.game_over = False

        self.reset()
        
        # Self-validation
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_lives = self.PLAYER_LIVES
        self.player_projectiles = []
        self.player_shoot_timer = 0
        self.player_shield_active = False
        self.player_shield_timer = 0
        self.player_shield_cooldown_timer = 0
        self.player_invincibility_timer = 0
        
        self.aliens = []
        self.alien_projectiles = []
        self.explosions = []

        self.score = 0
        self.steps = 0
        self.stage = 1
        self.aliens_destroyed_total = 0
        self.aliens_destroyed_this_stage = 0
        self.stage_timer = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self.game_over = False

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.stage_timer -= 1

        # --- Update game logic ---
        reward += self._update_player(action)
        reward += self._update_projectiles()
        reward += self._update_aliens()
        self._update_effects()
        
        # --- Handle progression ---
        if self.aliens_destroyed_this_stage >= self.ALIENS_PER_STAGE:
            reward += self._advance_stage()
        elif not self.aliens and self.aliens_destroyed_this_stage < self.ALIENS_PER_STAGE:
            self._spawn_wave()

        # --- Check termination conditions ---
        terminated = False
        if self.player_lives <= 0:
            terminated = True
            reward -= 100 # Loss penalty
            self.game_over = True
        elif self.aliens_destroyed_total >= self.ALIENS_PER_STAGE * self.TOTAL_STAGES:
            terminated = True
            reward += 100 # Win bonus
            self.game_over = True
        elif self.stage_timer <= 0:
            terminated = True
            reward -= 50 # Time out penalty
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.SCREEN_WIDTH - 20)
        self.player_pos[1] = np.clip(self.player_pos[1], 20, self.SCREEN_HEIGHT - 20)
        
        # Cooldowns
        if self.player_shoot_timer > 0: self.player_shoot_timer -= 1
        if self.player_shield_timer > 0: self.player_shield_timer -= 1
        if self.player_shield_cooldown_timer > 0: self.player_shield_cooldown_timer -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1
        if self.player_shield_timer <= 0: self.player_shield_active = False

        # Shield
        if shift_held and self.player_shield_cooldown_timer <= 0:
            # sfx: shield_activate.wav
            self.player_shield_active = True
            self.player_shield_timer = self.PLAYER_SHIELD_DURATION
            self.player_shield_cooldown_timer = self.PLAYER_SHIELD_COOLDOWN

        # Shooting
        if space_held and self.player_shoot_timer <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append([self.player_pos[0], self.player_pos[1] - 20])
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
        
        return 0

    def _update_projectiles(self):
        reward = 0
        # Player projectiles
        for p in self.player_projectiles[:]:
            p[1] -= self.PROJECTILE_SPEED
            if p[1] < 0:
                self.player_projectiles.remove(p)
                continue
            for a in self.aliens[:]:
                if math.hypot(p[0] - a['pos'][0], p[1] - a['pos'][1]) < 15:
                    # sfx: explosion.wav
                    self._create_explosion(a['pos'], self.COLOR_ALIEN)
                    self.aliens.remove(a)
                    if p in self.player_projectiles: self.player_projectiles.remove(p)
                    self.score += 10
                    self.aliens_destroyed_total += 1
                    self.aliens_destroyed_this_stage += 1
                    reward += 1
                    break
        
        # Alien projectiles
        for p in self.alien_projectiles[:]:
            p['pos'][1] += self.PROJECTILE_SPEED
            if p['pos'][1] > self.SCREEN_HEIGHT:
                self.alien_projectiles.remove(p)
            
            # Dodge reward logic
            if not p['dodged'] and p['pos'][1] > self.player_pos[1] - 10 and p['pos'][1] < self.player_pos[1] + 10:
                if abs(p['pos'][0] - self.player_pos[0]) > 20: # If it misses
                    reward += 0.1
                    p['dodged'] = True

            # Collision with player
            if math.hypot(p['pos'][0] - self.player_pos[0], p['pos'][1] - self.player_pos[1]) < 20:
                if self.player_shield_active:
                    # sfx: shield_deflect.wav
                    self._create_explosion(p['pos'], self.COLOR_PLAYER_SHIELD)
                    if p in self.alien_projectiles: self.alien_projectiles.remove(p)
                elif self.player_invincibility_timer <= 0:
                    # sfx: player_hit.wav
                    self._create_explosion(self.player_pos, self.COLOR_PLAYER)
                    self.player_lives -= 1
                    self.player_invincibility_timer = self.PLAYER_INVINCIBILITY_DURATION
                    reward -= 1
                    if p in self.alien_projectiles: self.alien_projectiles.remove(p)

        return reward

    def _update_aliens(self):
        alien_speed = 1.0 + (self.stage - 1) * 0.5 + (self.aliens_destroyed_this_stage // 10) * 0.05
        fire_chance = 0.002 + (self.stage - 1) * 0.001 + (self.aliens_destroyed_this_stage // 10) * 0.0002
        
        for a in self.aliens:
            a['pos'][1] += alien_speed
            a['pos'][0] = a['initial_x'] + math.sin(a['pos'][1] * 0.02) * 40
            
            if a['pos'][1] > self.SCREEN_HEIGHT + 20:
                # Alien reached bottom, penalize and remove
                self.player_lives -=1
                self.aliens.remove(a)
                continue

            if self.np_random.random() < fire_chance:
                # sfx: alien_shoot.wav
                self.alien_projectiles.append({'pos': list(a['pos']), 'dodged': False})
        return 0

    def _update_effects(self):
        # Explosions
        for e in self.explosions[:]:
            for p in e['particles']:
                p[0][0] += p[1][0]
                p[0][1] += p[1][1]
                p[2] -= 0.05 # Fade
            e['life'] -= 1
            if e['life'] <= 0:
                self.explosions.remove(e)
        
        # Starfield
        for star in self.stars:
            star[1] = (star[1] + star[3]) % self.SCREEN_HEIGHT
            if star[1] < star[3]: # Reset star at top if it scrolled off bottom
                star[0] = self.np_random.integers(0, self.SCREEN_WIDTH)

    def _spawn_wave(self):
        aliens_to_spawn = min(self.ALIENS_PER_WAVE, self.ALIENS_PER_STAGE - self.aliens_destroyed_this_stage)
        for i in range(aliens_to_spawn):
            row = i // 5
            col = i % 5
            x = self.SCREEN_WIDTH * 0.25 + col * (self.SCREEN_WIDTH * 0.5 / 4)
            y = -30 - row * 40
            self.aliens.append({'pos': [x, y], 'initial_x': x})

    def _advance_stage(self):
        reward = 2
        self.stage += 1
        if self.stage > self.TOTAL_STAGES:
            # This case is handled by win condition in step()
            return reward
        
        self.aliens_destroyed_this_stage = 0
        self.stage_timer = self.STAGE_TIME_LIMIT_SECONDS * self.FPS
        self._spawn_wave()
        return reward

    def _create_explosion(self, pos, base_color):
        num_particles = 20
        particles = []
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            velocity = [math.cos(angle) * speed, math.sin(angle) * speed]
            # Particles are [pos, vel, alpha]
            particles.append([list(pos), velocity, 1.0])
        self.explosions.append({'particles': particles, 'life': 20, 'color': base_color})

    def _get_observation(self):
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.stage,
            "aliens_left_in_stage": self.ALIENS_PER_STAGE - self.aliens_destroyed_this_stage
        }

    def _init_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.choice([1, 1, 1, 2, 2, 3])
            speed = 0.2 + (size * 0.2)
            self.stars.append([x, y, size, speed])
            
    def _render_game(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        for x, y, size, speed in self.stars:
            color = self.COLOR_STARS[size-1]
            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

        # Effects
        self._render_explosions()

        # Entities
        self._render_projectiles()
        self._render_aliens()
        self._render_player()
        
        # UI
        self._render_ui()

    def _render_player(self):
        pos = (int(self.player_pos[0]), int(self.player_pos[1]))
        
        # Invincibility blink
        if self.player_invincibility_timer > 0 and self.steps % 10 < 5:
            return

        # Shield
        if self.player_shield_active:
            radius = 30 + (self.PLAYER_SHIELD_DURATION - self.player_shield_timer) * 0.5
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(radius), self.COLOR_PLAYER_SHIELD)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], int(radius), self.COLOR_PLAYER_SHIELD)

        # Ship
        p1 = (pos[0], pos[1] - 18)
        p2 = (pos[0] - 12, pos[1] + 9)
        p3 = (pos[0] + 12, pos[1] + 9)
        pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
        
        # Engine flare
        flare_length = 8 + random.randint(0, 8)
        f1 = (pos[0] - 5, pos[1] + 10)
        f2 = (pos[0] + 5, pos[1] + 10)
        f3 = (pos[0], pos[1] + 10 + flare_length)
        pygame.gfxdraw.aapolygon(self.screen, (f1,f2,f3), self.COLOR_PLAYER_PROJECTILE)

    def _render_aliens(self):
        for a in self.aliens:
            pos = (int(a['pos'][0]), int(a['pos'][1]))
            p1 = (pos[0], pos[1] + 10)
            p2 = (pos[0] - 10, pos[1] - 5)
            p3 = (pos[0] + 10, pos[1] - 5)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_ALIEN)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pos = (int(p[0]), int(p[1]))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, (pos[0]-2, pos[1]-8, 4, 16))
        for p in self.alien_projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_PROJECTILE, pos, 5)

    def _render_explosions(self):
        for e in self.explosions:
            for p in e['particles']:
                pos = (int(p[0][0]), int(p[0][1]))
                alpha = max(0, int(p[2] * 255))
                color = (*e['color'], alpha)
                temp_surf = pygame.Surface((4,4), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (2,2), 2)
                self.screen.blit(temp_surf, (pos[0]-2, pos[1]-2))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_small.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 160, 15))
        for i in range(self.player_lives):
            pos = (self.SCREEN_WIDTH - 100 + i * 25, 22)
            p1 = (pos[0], pos[1] - 9)
            p2 = (pos[0] - 6, pos[1] + 4)
            p3 = (pos[0] + 6, pos[1] + 4)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

        # Stage and Timer
        stage_text = self.font_small.render(f"STAGE: {self.stage}/{self.TOTAL_STAGES}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.SCREEN_WIDTH / 2 - 100, 15))
        timer_text = self.font_small.render(f"TIME: {self.stage_timer // self.FPS}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH / 2 + 30, 15))

        # Game Over Text
        if self.game_over:
            if self.player_lives <= 0 or self.stage_timer <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            
            over_text = self.font_large.render(msg, True, (255, 255, 0))
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(over_text, text_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    
    # --- Human Player Controls ---
    # This setup allows for smooth human play by tracking key presses.
    # The agent's action is determined by the combination of keys held down.
    
    obs, info = env.reset()
    terminated = False
    
    # Key state dictionary
    keys_held = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False,
    }

    # Pygame setup for human play
    pygame.display.set_caption("Galactic Defender")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                    terminated = False
            elif event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # --- Action Mapping ---
        movement = 0 # no-op
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space_pressed = 1 if keys_held[pygame.K_SPACE] else 0
        shift_pressed = 1 if (keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT]) else 0
        
        action = [movement, space_pressed, shift_pressed]

        # --- Step Environment ---
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render ---
        # The observation is already a rendered frame, so we just need to display it.
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(env.FPS)

    env.close()