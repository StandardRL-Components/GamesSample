
# Generated: 2025-08-27T16:10:49.232937
# Source Brief: brief_01144.md
# Brief Index: 1144

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
    A retro-style, top-down arcade space shooter Gymnasium environment.
    The player controls a ship at the bottom of the screen and must defeat
    three waves of descending aliens.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Hold Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defeat waves of descending aliens in this fast-paced arcade shooter before they destroy your ship."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 1500  # Extended from 1000 to allow for longer gameplay
        self.TOTAL_WAVES = 3

        # Player settings
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_RATE = 5  # Cooldown in frames

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_GLOW = (50, 255, 50, 50)
        self.COLOR_ALIEN_1 = (255, 50, 50)
        self.COLOR_ALIEN_2 = (255, 150, 50)
        self.COLOR_ALIEN_3 = (200, 50, 255)
        self.COLOR_PLAYER_PROJ = (200, 255, 255)
        self.COLOR_ALIEN_PROJ = (255, 200, 50)
        self.COLOR_EXPLOSION = (255, 255, 0)
        self.COLOR_UI = (220, 220, 220)
        self.COLOR_STAR = (180, 180, 200)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Internal state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.player_pos = pygame.Vector2(0, 0)
        self.player_lives = 0
        self.player_projectiles = []
        self.alien_projectiles = []
        self.aliens = []
        self.explosions = []
        self.stars = []
        self.current_wave = 0
        self.player_shoot_cooldown = 0
        self.player_invincibility_timer = 0
        self.player_size = pygame.Vector2(30, 20)

        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = 3
        self.player_shoot_cooldown = 0
        self.player_invincibility_timer = 0

        self.player_projectiles.clear()
        self.alien_projectiles.clear()
        self.aliens.clear()
        self.explosions.clear()
        
        self.current_wave = 0
        self._spawn_wave()
        
        if not self.stars:
            self._create_stars()

        return self._get_observation(), self._get_info()

    def _create_stars(self):
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(100)
        ]

    def _spawn_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            self.game_won = True
            return

        rows = 2 + self.current_wave
        cols = 8
        h_spacing = self.WIDTH * 0.8 / cols
        v_spacing = 40
        start_x = self.WIDTH * 0.1
        start_y = 50

        for row in range(rows):
            for col in range(cols):
                alien_type = random.randint(1, min(self.current_wave, 3))
                pos = pygame.Vector2(start_x + col * h_spacing, start_y + row * v_spacing)
                
                # Alien properties based on wave
                speed_multiplier = 1 + (self.current_wave - 1) * 0.2
                fire_rate_multiplier = 1 + (self.current_wave - 1) * 0.2

                alien = {
                    "pos": pos,
                    "type": alien_type,
                    "size": 25,
                    "speed": 0.5 * speed_multiplier,
                    "fire_rate": 150 / fire_rate_multiplier, # Lower is faster
                    "fire_cooldown": random.randint(0, 150),
                    "move_pattern": "sine" if alien_type > 1 else "linear",
                    "move_phase": random.uniform(0, 2 * math.pi),
                    "origin_x": pos.x
                }
                self.aliens.append(alien)

    def step(self, action):
        reward = 0.0
        terminated = False
        
        if not self.game_over:
            reward += 0.01  # Small survival reward per frame

            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            
            self._handle_player_input(movement, space_held)
            self._update_game_state()
            reward += self._handle_collisions()
            
            if not self.aliens and not self.game_won:
                reward += 100  # Wave clear reward
                self._spawn_wave()
                if self.game_won:
                    reward += 500 # Game win reward
            
            self.steps += 1

        terminated = self.player_lives <= 0 or self.steps >= self.MAX_STEPS or self.game_won
        if terminated:
            self.game_over = True

        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Player Movement
        if movement == 3:  # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, self.player_size.x / 2, self.WIDTH - self.player_size.x / 2)

        # Player Shooting
        self.player_shoot_cooldown = max(0, self.player_shoot_cooldown - 1)
        if space_held and self.player_shoot_cooldown == 0:
            # Sound effect placeholder: # pew pew
            proj_pos = self.player_pos + pygame.Vector2(0, -self.player_size.y / 2)
            self.player_projectiles.append(proj_pos)
            self.player_shoot_cooldown = self.PLAYER_FIRE_RATE

    def _update_game_state(self):
        # Player invincibility
        self.player_invincibility_timer = max(0, self.player_invincibility_timer - 1)

        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= 12
            if proj.y < 0:
                self.player_projectiles.remove(proj)

        # Alien movement and shooting
        for alien in self.aliens:
            alien['pos'].y += alien['speed']
            if alien['move_pattern'] == 'sine':
                alien['move_phase'] += 0.05
                alien['pos'].x = alien['origin_x'] + math.sin(alien['move_phase']) * 40
            
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                # Sound effect placeholder: # alien zap
                proj_pos = alien['pos'] + pygame.Vector2(0, alien['size'] / 2)
                self.alien_projectiles.append(proj_pos)
                alien['fire_cooldown'] = alien['fire_rate'] + random.uniform(-20, 20)

        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj.y += 6
            if proj.y > self.HEIGHT:
                self.alien_projectiles.remove(proj)
                # reward += 0.5 # Dodge reward - can be noisy, so disabled for now

        # Explosions
        for exp in self.explosions[:]:
            exp['radius'] += exp['speed']
            exp['alpha'] = max(0, exp['alpha'] - 10)
            if exp['alpha'] == 0:
                self.explosions.remove(exp)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.player_size.x / 2, self.player_pos.y - self.player_size.y / 2, self.player_size.x, self.player_size.y)

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if (proj - alien['pos']).length() < alien['size'] / 2:
                    # Sound effect placeholder: # alien explosion
                    self._create_explosion(alien['pos'], 40, 3)
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self.score += 100
                    reward += 10
                    break

        # Alien projectiles vs Player
        if self.player_invincibility_timer == 0:
            for proj in self.alien_projectiles[:]:
                if player_rect.collidepoint(proj):
                    self.alien_projectiles.remove(proj)
                    reward += self._hit_player()
                    break

        # Aliens vs Player
        if self.player_invincibility_timer == 0:
            for alien in self.aliens[:]:
                if player_rect.colliderect(alien['pos'].x - alien['size']/2, alien['pos'].y - alien['size']/2, alien['size'], alien['size']):
                    self._create_explosion(alien['pos'], 40, 3)
                    self.aliens.remove(alien)
                    reward += self._hit_player()
                    break

        # Aliens reaching bottom
        for alien in self.aliens[:]:
            if alien['pos'].y > self.HEIGHT - 20:
                self.aliens.remove(alien)
                reward += self._hit_player() # Penalize as a hit

        return reward

    def _hit_player(self):
        # Sound effect placeholder: # player hit
        self.player_lives -= 1
        self._create_explosion(self.player_pos, 60, 4)
        self.player_invincibility_timer = 90  # 3 seconds of invincibility
        return -50 # Penalty for getting hit

    def _create_explosion(self, pos, max_radius, speed):
        self.explosions.append({
            "pos": pos.copy(),
            "radius": 0,
            "max_radius": max_radius,
            "speed": speed,
            "alpha": 255
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.rect(self.screen, self.COLOR_STAR, (x, y, size, size))

        # Explosions
        for exp in self.explosions:
            color = (*self.COLOR_EXPLOSION, exp['alpha'])
            pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'].x), int(exp['pos'].y), int(exp['radius']), color)
            
        # Player projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (int(proj.x) - 2, int(proj.y) - 5, 4, 10))

        # Alien projectiles
        for proj in self.alien_projectiles:
            pygame.gfxdraw.filled_trigon(self.screen, int(proj.x), int(proj.y+4), int(proj.x-4), int(proj.y-4), int(proj.x+4), int(proj.y-4), self.COLOR_ALIEN_PROJ)

        # Aliens
        for alien in self.aliens:
            x, y, size = int(alien['pos'].x), int(alien['pos'].y), alien['size']
            rect = pygame.Rect(x - size/2, y - size/2, size, size)
            if alien['type'] == 1: color = self.COLOR_ALIEN_1
            elif alien['type'] == 2: color = self.COLOR_ALIEN_2
            else: color = self.COLOR_ALIEN_3
            pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Player
        if self.player_lives > 0:
            is_invincible = self.player_invincibility_timer > 0
            if not (is_invincible and (self.steps // 3) % 2 == 0):
                px, py = int(self.player_pos.x), int(self.player_pos.y)
                pw, ph = self.player_size.x, self.player_size.y
                p1 = (px, py - ph / 2)
                p2 = (px - pw / 2, py + ph / 2)
                p3 = (px + pw / 2, py + ph / 2)
                # Glow effect
                pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER_GLOW)
                # Main ship
                pygame.gfxdraw.aatrigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)
                pygame.gfxdraw.filled_trigon(self.screen, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), int(p3[0]), int(p3[1]), self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Lives
        lives_surf = self.font_ui.render(f"LIVES: {self.player_lives}", True, self.COLOR_UI)
        self.screen.blit(lives_surf, (self.WIDTH - lives_surf.get_width() - 10, 10))

        # Wave
        wave_surf = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_surf, (self.WIDTH / 2 - wave_surf.get_width() / 2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER"
            if self.game_won:
                msg = "YOU WIN!"
            
            over_surf = self.font_game_over.render(msg, True, self.COLOR_UI)
            over_rect = over_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(over_surf, over_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
            "game_won": self.game_won
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    clock = pygame.time.Clock()

    terminated = False
    total_reward = 0

    print("--- Game Start ---")
    print(GameEnv.user_guide)

    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

    print(f"--- Game Over ---")
    print(f"Final Score: {info['score']}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Steps: {info['steps']}")
    if info['game_won']:
        print("Result: VICTORY!")
    else:
        print("Result: DEFEAT")

    env.close()