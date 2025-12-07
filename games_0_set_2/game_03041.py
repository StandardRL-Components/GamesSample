
# Generated: 2025-08-28T06:48:23.573589
# Source Brief: brief_03041.md
# Brief Index: 3041

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: ←→ to move ship. Hold space to fire your weapon. Dodge alien fire!"
    )

    # User-facing description of the game
    game_description = (
        "Defend Earth from descending alien invaders in this retro-inspired arcade shooter. Clear all three waves to win!"
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Colors (Bright for interactive, dark for background)
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PLAYER_FLAME = (255, 150, 0)
        self.COLOR_ALIEN_1 = (255, 50, 50)
        self.COLOR_ALIEN_2 = (255, 150, 50)
        self.COLOR_ALIEN_3 = (255, 200, 50)
        self.COLOR_BULLET_PLAYER = (150, 255, 255)
        self.COLOR_BULLET_ALIEN = (255, 100, 100)
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_UI_TEXT = (240, 240, 240)
        
        # Fonts
        try:
            self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
            self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        except pygame.error:
            self.font_main = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 52)
            
        # Game constants
        self.MAX_STEPS = 10000
        self.PLAYER_SPEED = 6
        self.PLAYER_SHOOT_COOLDOWN = 6 # 5 shots per second at 30fps
        self.MAX_WAVES = 3
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_shoot_timer = None
        self.player_bullets = None
        self.aliens = None
        self.alien_bullets = None
        self.alien_move_direction = None
        self.alien_move_timer = None
        self.alien_drop_dist = None
        self.particles = None
        self.stars = None
        self.current_wave = None
        self.steps = None
        self.score = None
        self.high_score = 0
        self.game_over = None
        self.game_won = None
        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = 3
        self.player_shoot_timer = 0
        
        self.player_bullets = []
        self.alien_bullets = []
        self.particles = []
        
        self.current_wave = 1
        self._spawn_wave()
        
        if self.stars is None:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(150)
            ]
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0
        terminated = False

        if not self.game_over:
            # Survival reward
            reward += 0.01

            # Handle player input
            self._handle_input(action)
            
            # Update game state
            self._update_player()
            reward += self._update_bullets()
            reward += self._update_aliens()
            self._update_particles()
            
            # Check for wave clear
            if not self.aliens and not self.game_won:
                reward += 100 # Wave clear bonus
                self.current_wave += 1
                if self.current_wave > self.MAX_WAVES:
                    self.game_won = True
                    self.game_over = True
                    reward += 500 # Game win bonus
                else:
                    self._spawn_wave()
                    # sfx: wave_clear.wav

        # Check for termination conditions
        if self.player_lives <= 0:
            if not self.game_over: # Only apply penalty once
                reward -= 100 # Game over penalty
                # sfx: game_over.wav
            self.game_over = True
            self.game_won = False

        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.steps += 1
        self.high_score = max(self.high_score, self.score)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action

        # Movement (0=none, 1=up, 2=down, 3=left, 4=right)
        # In this game, up/down movement is ignored.
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = max(20, min(self.WIDTH - 20, self.player_pos.x))
        
        # Shooting
        if space_held and self.player_shoot_timer == 0:
            self.player_bullets.append(pygame.Vector2(self.player_pos.x, self.player_pos.y - 20))
            self.player_shoot_timer = self.PLAYER_SHOOT_COOLDOWN
            # sfx: player_shoot.wav

    def _update_player(self):
        if self.player_shoot_timer > 0:
            self.player_shoot_timer -= 1
            
    def _update_bullets(self):
        reward = 0
        
        # Player bullets
        for bullet in self.player_bullets[:]:
            bullet.y -= 12
            if bullet.y < 0:
                self.player_bullets.remove(bullet)
                reward -= 0.2 # Penalty for missing
                continue
            
            for alien in self.aliens[:]:
                if pygame.Rect(alien['pos'][0]-alien['size']/2, alien['pos'][1]-alien['size']/2, alien['size'], alien['size']).collidepoint(bullet):
                    self._create_explosion(alien['pos'], alien['color'], 20)
                    self.aliens.remove(alien)
                    self.player_bullets.remove(bullet)
                    reward += 10 # Reward for destroying alien
                    self.score += 100
                    # sfx: alien_destroyed.wav
                    break

        # Alien bullets
        for bullet in self.alien_bullets[:]:
            bullet['pos'].y += bullet['speed']
            if bullet['pos'].y > self.HEIGHT:
                self.alien_bullets.remove(bullet)
                continue
            
            player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 10, 30, 20)
            if player_rect.collidepoint(bullet['pos']):
                self.alien_bullets.remove(bullet)
                self.player_lives -= 1
                self._create_explosion(self.player_pos, self.COLOR_PLAYER, 40)
                reward -= 5 # Penalty for getting hit
                # sfx: player_hit.wav
                break
                
        return reward

    def _update_aliens(self):
        self.alien_move_timer -= 1
        move_sideways = False
        move_down = False
        
        if self.alien_move_timer <= 0:
            self.alien_move_timer = self.alien_move_cooldown
            move_sideways = True
        
        if not self.aliens:
            return 0
            
        # Check if any alien hits the side
        for alien in self.aliens:
            if move_sideways:
                if (alien['pos'].x >= self.WIDTH - 20 and self.alien_move_direction > 0) or \
                   (alien['pos'].x <= 20 and self.alien_move_direction < 0):
                    self.alien_move_direction *= -1
                    move_down = True
                    break
        
        # Update alien positions and handle shooting
        for alien in self.aliens:
            if move_sideways:
                alien['pos'].x += self.alien_move_direction * 20
            if move_down:
                alien['pos'].y += self.alien_drop_dist
            
            # Animation
            alien['anim_timer'] = (alien['anim_timer'] + 1) % 20

            # Check for game over by aliens reaching bottom
            if alien['pos'].y > self.HEIGHT - 50:
                self.player_lives = 0
                return 0

            # Shooting logic
            fire_chance = self.alien_fire_rate / len(self.aliens) if self.aliens else 0
            if self.np_random.random() < fire_chance:
                bullet_speed = 4 + self.current_wave * 0.5
                self.alien_bullets.append({'pos': pygame.Vector2(alien['pos']), 'speed': bullet_speed})
                # sfx: alien_shoot.wav
        
        return 0

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_wave(self):
        self.aliens = []
        rows, cols = 2 + self.current_wave, 8
        alien_types = [self.COLOR_ALIEN_1, self.COLOR_ALIEN_2, self.COLOR_ALIEN_3]
        
        for r in range(rows):
            for c in range(cols):
                alien_type = min(r, len(alien_types)-1)
                pos = pygame.Vector2(100 + c * 60, 50 + r * 40)
                self.aliens.append({
                    'pos': pos,
                    'color': alien_types[alien_type],
                    'size': 20,
                    'anim_timer': self.np_random.integers(0, 20)
                })
        
        self.alien_move_direction = 1
        self.alien_move_cooldown = max(15 - self.current_wave * 2, 5)
        self.alien_move_timer = self.alien_move_cooldown
        self.alien_drop_dist = 10
        self.alien_fire_rate = 0.005 + 0.005 * self.current_wave

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 + 1
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(15, 30),
                'color': color,
                'size': self.np_random.integers(2, 5)
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
            
        # Player
        if self.player_lives > 0:
            self._render_player()
            
        # Bullets
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_BULLET_PLAYER, (int(bullet.x-2), int(bullet.y-5), 4, 10))
        for bullet in self.alien_bullets:
            pygame.gfxdraw.filled_circle(self.screen, int(bullet['pos'].x), int(bullet['pos'].y), 4, self.COLOR_BULLET_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, int(bullet['pos'].x), int(bullet['pos'].y), 4, self.COLOR_BULLET_ALIEN)

        # Aliens
        for alien in self.aliens:
            self._render_alien(alien)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 30))
            color = (min(255, p['color'][0]+alpha), min(255, p['color'][1]+alpha), min(255, p['color'][2]+alpha))
            pygame.draw.circle(self.screen, color, (int(p['pos'].x), int(p['pos'].y)), int(p['size']))

    def _render_player(self):
        p = self.player_pos
        points = [(p.x, p.y - 15), (p.x - 15, p.y + 10), (p.x + 15, p.y + 10)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
        # Engine flame
        flame_height = 10 + (self.steps % 5) # Flicker effect
        flame_points = [(p.x, p.y + 12), (p.x - 6, p.y + 10), (p.x + 6, p.y + 10)]
        flame_points[0] = (p.x, p.y + 10 + flame_height)
        pygame.gfxdraw.aapolygon(self.screen, flame_points, self.COLOR_PLAYER_FLAME)
        pygame.gfxdraw.filled_polygon(self.screen, flame_points, self.COLOR_PLAYER_FLAME)

    def _render_alien(self, alien):
        pos = alien['pos']
        size = alien['size']
        color = alien['color']
        anim_offset = abs(10 - alien['anim_timer']) / 10 * 4 # Bobbing effect
        
        rect = pygame.Rect(pos.x - size/2, pos.y - size/2 + anim_offset, size, size)
        pygame.draw.rect(self.screen, color, rect, border_radius=3)
        
        eye_size = 3
        pygame.draw.rect(self.screen, self.COLOR_BG, (rect.left + 4, rect.top + 5, eye_size, eye_size))
        pygame.draw.rect(self.screen, self.COLOR_BG, (rect.right - 4 - eye_size, rect.top + 5, eye_size, eye_size))

    def _render_ui(self):
        # Score and High Score
        score_text = self.font_main.render(f"SCORE: {self.score:06d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_main.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 180, 10))
        for i in range(self.player_lives):
            ship_points = [(self.WIDTH - 90 + i*30, 20), (self.WIDTH - 100 + i*30, 30), (self.WIDTH - 80 + i*30, 30)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, ship_points)

        # Wave
        wave_text = self.font_main.render(f"WAVE {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        text_rect = wave_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT - 20))
        self.screen.blit(wave_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER" if not self.game_won else "YOU WIN!"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            pygame.draw.rect(self.screen, self.COLOR_BG, end_rect.inflate(20, 20))
            self.screen.blit(end_text, end_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "high_score": self.high_score,
            "steps": self.steps,
            "player_lives": self.player_lives,
            "current_wave": self.current_wave
        }
    
    def close(self):
        pygame.quit()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a different setup
    pygame.display.set_caption("Arcade Shooter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and (terminated or truncated):
                obs, info = env.reset()

        # Control frame rate
        env.clock.tick(30)
        
    env.close()