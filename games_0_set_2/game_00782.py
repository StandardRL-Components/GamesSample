
# Generated: 2025-08-27T14:46:17.020905
# Source Brief: brief_00782.md
# Brief Index: 782

        
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
    A Gymnasium environment for a retro-inspired arcade shooter.
    The player controls a ship at the bottom of the screen, moving left and right,
    and must shoot down waves of descending aliens.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move. Press space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from descending waves of procedurally generated alien invaders in this retro-inspired arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 3000
        self.MAX_WAVES = 5

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
        self.font_small = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("consolas", 48, bold=True)

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ALIEN_1 = (255, 60, 60)
        self.COLOR_ALIEN_2 = (200, 60, 255)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 255)
        self.COLOR_PROJECTILE_ALIEN = (255, 100, 100)
        self.COLOR_EXPLOSION_1 = (255, 200, 0)
        self.COLOR_EXPLOSION_2 = (255, 100, 0)
        self.COLOR_TEXT = (220, 220, 220)
        
        # Game state variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.win = None
        self.current_wave = None
        self.player_pos_x = None
        self.player_fire_cooldown = None
        self.prev_space_held = None
        self.aliens = None
        self.projectiles = None
        self.explosions = None
        self.stars = None
        
        # Create persistent starfield
        self.stars = [
            [random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.choice([1, 2, 3])]
            for _ in range(150)
        ]
        
        # Initialize state variables
        self.reset()
        
        # Self-validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.current_wave = 1
        
        self.player_pos_x = self.WIDTH // 2
        self.player_fire_cooldown = 0
        self.prev_space_held = False
        
        self.aliens = []
        self.projectiles = []
        self.explosions = []

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        # shift_held is unused in this implementation

        self.steps += 1
        reward += 0.1  # Continuous reward for each frame survived

        # Update game logic
        self._update_cooldowns()
        self._handle_player_input(movement, space_held)
        self._update_aliens()
        self._update_projectiles()
        self._update_explosions()
        self._update_stars()

        # Handle collisions and collect event-based rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # Check for wave completion and assign goal-oriented rewards
        if not self.aliens and not self.game_over:
            self.current_wave += 1
            if self.current_wave > self.MAX_WAVES:
                self.win = True
                self.game_over = True
                reward += 500  # Win game reward
            else:
                reward += 100  # Wave complete reward
                self._spawn_wave()
        
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        # Player movement
        if movement == 3:  # Left
            self.player_pos_x -= 6
        elif movement == 4: # Right
            self.player_pos_x += 6
        self.player_pos_x = max(20, min(self.WIDTH - 20, self.player_pos_x))

        # Player firing (on key press, not hold)
        if space_held and not self.prev_space_held and self.player_fire_cooldown <= 0:
            # SFX: Player shoot
            self.projectiles.append({
                "pos": [self.player_pos_x, self.HEIGHT - 40],
                "vel": -12,
                "type": "player",
                "size": (4, 12)
            })
            self.player_fire_cooldown = 8 # Cooldown in frames
        self.prev_space_held = space_held
    
    def _update_cooldowns(self):
        if self.player_fire_cooldown > 0:
            self.player_fire_cooldown -= 1

    def _spawn_wave(self):
        self.aliens.clear()
        rows, cols = 4, 10
        x_spacing, y_spacing = 50, 40
        x_offset = (self.WIDTH - (cols - 1) * x_spacing) // 2
        y_offset = 50
        
        for r in range(rows):
            for c in range(cols):
                pattern = 'sine' if r % 2 == 0 else 'zigzag'
                alien_type = 1 if r < 2 else 2
                self.aliens.append({
                    "initial_pos": [x_offset + c * x_spacing, y_offset + r * y_spacing],
                    "pos": [x_offset + c * x_spacing, y_offset + r * y_spacing],
                    "pattern": pattern,
                    "type": alien_type,
                    "size": (30, 20)
                })

    def _update_aliens(self):
        # Difficulty scaling
        wave_speed = 0.5 + (self.current_wave - 1) * 0.2
        fire_prob = 0.01 + (self.current_wave - 1) * 0.005
        
        for alien in self.aliens:
            # Vertical movement (descent)
            alien['initial_pos'][1] += wave_speed * 0.2
            
            # Horizontal movement
            amplitude = 40
            frequency = 0.02
            if alien['pattern'] == 'sine':
                offset = amplitude * math.sin(frequency * self.steps + alien['initial_pos'][0])
            else: # zigzag
                period = 2 / frequency
                offset = amplitude * (2 * abs(round(((self.steps + alien['initial_pos'][0]) % period) / period - 0.5)) - 0.5) * 2
            alien['pos'][0] = alien['initial_pos'][0] + offset
            alien['pos'][1] = alien['initial_pos'][1]

            # Alien firing
            if self.aliens and self.np_random.random() < fire_prob / len(self.aliens):
                # SFX: Alien shoot
                self.projectiles.append({
                    "pos": [alien['pos'][0], alien['pos'][1] + alien['size'][1] // 2],
                    "vel": 6,
                    "type": "alien",
                    "size": (4, 8)
                })

            # Check if aliens reached player's vertical level
            if alien['pos'][1] > self.HEIGHT - 50:
                self.game_over = True

    def _update_projectiles(self):
        self.projectiles = [
            p for p in self.projectiles if 0 < p['pos'][1] + p['vel'] < self.HEIGHT
        ]
        for p in self.projectiles:
            p['pos'][1] += p['vel']

    def _update_explosions(self):
        self.explosions = [e for e in self.explosions if e['life'] > 0]
        for e in self.explosions:
            e['life'] -= 1
            e['radius'] += 1

    def _update_stars(self):
        for star in self.stars:
            star[1] += star[2] * 0.5 # Move star down for parallax effect
            if star[1] > self.HEIGHT:
                star[0] = random.randint(0, self.WIDTH)
                star[1] = 0

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos_x - 15, self.HEIGHT - 40, 30, 20)
        
        # Player projectiles vs Aliens
        remaining_aliens = []
        for alien in self.aliens:
            alien_hit = False
            alien_rect = pygame.Rect(alien['pos'][0] - alien['size'][0]//2, alien['pos'][1] - alien['size'][1]//2, *alien['size'])
            
            non_colliding_projectiles = []
            for p in self.projectiles:
                if p['type'] == 'player':
                    proj_rect = pygame.Rect(p['pos'][0] - p['size'][0]//2, p['pos'][1] - p['size'][1]//2, *p['size'])
                    if alien_rect.colliderect(proj_rect):
                        # SFX: Explosion
                        self.explosions.append({'pos': alien['pos'][:], 'radius': 5, 'life': 15})
                        self.score += 10
                        reward += 1  # Reward for destroying an alien
                        alien_hit = True
                    else:
                        non_colliding_projectiles.append(p)
                else:
                    non_colliding_projectiles.append(p)
            self.projectiles = non_colliding_projectiles
            
            if not alien_hit:
                remaining_aliens.append(alien)
        self.aliens = remaining_aliens

        # Alien projectiles vs Player
        for p in self.projectiles:
            if p['type'] == 'alien':
                proj_rect = pygame.Rect(p['pos'][0] - p['size'][0]//2, p['pos'][1] - p['size'][1]//2, *p['size'])
                if player_rect.colliderect(proj_rect):
                    self.game_over = True
                    reward -= 10  # Penalty for being hit
                    # SFX: Player hit
                    self.explosions.append({'pos': [self.player_pos_x, self.HEIGHT-30], 'radius': 10, 'life': 30})
                    break
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
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
            "wave": self.current_wave,
            "game_over": self.game_over,
        }

    def _render_game(self):
        # Draw stars
        for x, y, r in self.stars:
            color_val = 50 + r * 30
            pygame.draw.circle(self.screen, (color_val, color_val, color_val), (int(x), int(y)), r)

        # Draw projectiles
        for p in self.projectiles:
            color = self.COLOR_PROJECTILE_PLAYER if p['type'] == 'player' else self.COLOR_PROJECTILE_ALIEN
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = p['size']
            rect = pygame.Rect(pos[0] - size[0]//2, pos[1] - size[1]//2, size[0], size[1])
            pygame.draw.rect(self.screen, color, rect, border_radius=2)
            
        # Draw aliens
        for alien in self.aliens:
            self._draw_alien(alien)

        # Draw player
        if not self.game_over or self.win:
            self._draw_player()
            
        # Draw explosions
        for e in self.explosions:
            alpha = max(0, 255 * (e['life'] / 15))
            s = pygame.Surface((e['radius']*2, e['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_EXPLOSION_1, alpha), (e['radius'], e['radius']), e['radius'])
            s2_rad = max(0, e['radius'] * 0.5)
            pygame.draw.circle(s, (*self.COLOR_EXPLOSION_2, alpha), (e['radius'], e['radius']), int(s2_rad))
            self.screen.blit(s, (int(e['pos'][0] - e['radius']), int(e['pos'][1] - e['radius'])))

    def _draw_player(self):
        x = int(self.player_pos_x)
        y = self.HEIGHT - 30
        points = [(x, y - 15), (x - 15, y + 10), (x + 15, y + 10)]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        pygame.draw.lines(self.screen, (200, 255, 220), True, points, 2)
        # Engine flare
        if self.player_fire_cooldown < 3: # Flicker when ready to fire
            flare_height = 5 + random.randint(0, 5)
            flare_points = [(x-7, y+10), (x+7, y+10), (x, y+10+flare_height)]
            pygame.draw.polygon(self.screen, (255, 255, 150), flare_points)

    def _draw_alien(self, alien):
        x, y = int(alien['pos'][0]), int(alien['pos'][1])
        w, h = alien['size']
        color = self.COLOR_ALIEN_1 if alien['type'] == 1 else self.COLOR_ALIEN_2
        body = pygame.Rect(x - w/2, y - h/2, w, h)
        pygame.draw.rect(self.screen, color, body, border_radius=4)
        # Eyes
        eye_y = y - h/4
        pygame.draw.circle(self.screen, (255,255,255), (int(x - w/4), int(eye_y)), 3)
        pygame.draw.circle(self.screen, (255,255,255), (int(x + w/4), int(eye_y)), 3)

    def _render_ui(self):
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))

        if self.game_over:
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN_1
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        """ Call this at the end of __init__ to verify implementation. """
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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows for manual play and testing of the environment.
    env = GameEnv()
    
    # --- Manual Play Example ---
    obs, info = env.reset()
    done = False
    
    # Create a window for display if not running headless
    try:
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        pygame.display.set_caption("Galactic Defender")
        clock = pygame.time.Clock()
        is_headless = False
    except pygame.error:
        print("Pygame display unavailable. Running in headless mode.")
        is_headless = True

    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        if not is_headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action[0] = 3
            if keys[pygame.K_RIGHT]:
                action[0] = 4
            if keys[pygame.K_SPACE]:
                action[1] = 1
        else: # If headless, use random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not is_headless:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(30) # Limit to 30 FPS
        
        if done:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            if not is_headless:
                pygame.time.wait(2000)
    
    env.close()