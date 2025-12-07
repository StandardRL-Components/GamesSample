
# Generated: 2025-08-27T20:25:26.024660
# Source Brief: brief_02458.md
# Brief Index: 2458

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro vector-graphics shooter. Survive 5 waves of alien invaders and aim for a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 3000 # Increased for 5 waves
        self.MAX_WAVES = 5
        self.INITIAL_LIVES = 3

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_ALIEN_GLOW = (255, 50, 50, 40)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 180, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_STAR = (100, 100, 120)

        # Player settings
        self.PLAYER_SPEED = 6.0
        self.PLAYER_FRICTION = 0.9
        self.PLAYER_FIRE_COOLDOWN = 5  # frames
        self.PLAYER_INVULNERABILITY_DURATION = 60 # frames

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
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_wave = pygame.font.Font(pygame.font.get_default_font(), 24)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 18)
            self.font_wave = pygame.font.SysFont("monospace", 24)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.player_lives = 0
        self.player_fire_cooldown_timer = 0
        self.player_invulnerability_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []
        self.stars = []
        
        self.steps = 0
        self.score = 0
        self.current_wave = 0
        self.wave_clear_timer = 0
        self.game_over = False
        
        # Initialize state variables properly
        self.reset()
        
        # Run validation
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.current_wave = 1
        self.game_over = False
        self.wave_clear_timer = 0
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_lives = self.INITIAL_LIVES
        self.player_fire_cooldown_timer = 0
        self.player_invulnerability_timer = 0

        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.particles = []

        self._spawn_wave()
        self._generate_stars()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Cost of living
        
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        # Handle player input and movement
        self._handle_input(movement, space_held)
        self._update_player()

        # Update game objects if not in a "wave clear" pause
        if self.wave_clear_timer <= 0:
            reward += self._update_aliens()
            self._update_projectiles()
        else:
            self.wave_clear_timer -= 1
        
        self._update_particles()
        
        # Handle collisions and calculate rewards
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # Check for wave completion
        if not self.aliens and not self.game_over and self.wave_clear_timer <= 0:
            reward += 10
            self.score += 10
            self.current_wave += 1
            if self.current_wave > self.MAX_WAVES:
                reward += 100
                self.game_over = True
            else:
                self.wave_clear_timer = self.FPS * 2 # 2 second pause
                self._spawn_wave()

        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        if movement == 1: self.player_vel.y -= 1
        if movement == 2: self.player_vel.y += 1
        if movement == 3: self.player_vel.x -= 1
        if movement == 4: self.player_vel.x += 1
        
        if self.player_vel.length() > 1:
            self.player_vel.normalize_ip()

        if space_held and self.player_fire_cooldown_timer <= 0:
            # SFX: Player shoot
            self.player_projectiles.append(pygame.Vector2(self.player_pos.x, self.player_pos.y - 15))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player(self):
        self.player_pos += self.player_vel * self.PLAYER_SPEED
        self.player_vel *= self.PLAYER_FRICTION
        
        self.player_pos.x = np.clip(self.player_pos.x, 10, self.WIDTH - 10)
        self.player_pos.y = np.clip(self.player_pos.y, self.HEIGHT/2, self.HEIGHT - 20)

        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        if self.player_invulnerability_timer > 0:
            self.player_invulnerability_timer -= 1
            
    def _update_aliens(self):
        reward = 0
        for alien in self.aliens:
            # Move alien
            alien['pos'].x = alien['initial_pos'].x + math.sin(self.steps * alien['wave_speed'] + alien['phase']) * alien['amplitude']
            alien['pos'].y += alien['descent_speed']
            
            # Alien firing
            if self.np_random.random() < alien['fire_rate']:
                self.alien_projectiles.append(pygame.Vector2(alien['pos']))

            # Check if alien reached bottom (soft-lock prevention)
            if alien['pos'].y > self.HEIGHT - 10:
                self.aliens.remove(alien)
                self._player_hit()
                reward -= 5 # Penalty for letting one through
        return reward

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= 12
            if proj.y < 0:
                self.player_projectiles.remove(proj)
            else:
                self._create_particle(proj, self.COLOR_PROJECTILE, 1, -1, 5, 0.2)

        # Alien projectiles
        speed = 3 + (self.current_wave * 0.5)
        for proj in self.alien_projectiles[:]:
            proj.y += speed
            if proj.y > self.HEIGHT:
                self.alien_projectiles.remove(proj)
            else:
                self._create_particle(proj, self.COLOR_ALIEN, 1, 1, 5, 0.2)
                
    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] <= 0 or p['radius'] < 0.5:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.distance_to(alien['pos']) < 15:
                    reward += 0.1 # Hit reward
                    reward += 1.0 # Kill reward
                    self.score += 1
                    self._create_explosion(alien['pos'], self.COLOR_EXPLOSION, 20)
                    # SFX: Alien explosion
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    break

        # Alien projectiles vs player
        if self.player_invulnerability_timer <= 0:
            player_rect = pygame.Rect(self.player_pos.x - 7, self.player_pos.y - 7, 14, 14)
            for proj in self.alien_projectiles[:]:
                if player_rect.collidepoint(proj):
                    self._player_hit()
                    self.alien_projectiles.remove(proj)
                    break
        return reward
    
    def _player_hit(self):
        if self.player_invulnerability_timer > 0:
            return
        self.player_lives -= 1
        self.player_invulnerability_timer = self.PLAYER_INVULNERABILITY_DURATION
        self._create_explosion(self.player_pos, self.COLOR_PLAYER, 30)
        # SFX: Player hit/explosion
        if self.player_lives <= 0:
            self.game_over = True

    def _check_termination(self):
        return self.game_over or self.steps >= self.MAX_STEPS

    def _spawn_wave(self):
        num_aliens = 8 + self.current_wave * 2
        rows = 2 + self.current_wave // 2
        cols = (num_aliens // rows)
        
        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            x = self.WIDTH * (col + 1) / (cols + 1)
            y = 50 + row * 40
            
            alien = {
                'initial_pos': pygame.Vector2(x, y),
                'pos': pygame.Vector2(x, y),
                'wave_speed': 0.02 + self.current_wave * 0.005,
                'amplitude': 50 + self.current_wave * 10,
                'phase': self.np_random.random() * math.pi * 2,
                'descent_speed': 0.1 + self.current_wave * 0.02,
                'fire_rate': 0.002 + self.current_wave * 0.0005
            }
            self.aliens.append(alien)
    
    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append((
                self.np_random.integers(0, self.WIDTH),
                self.np_random.integers(0, self.HEIGHT),
                self.np_random.choice([1, 2, 3]) # size/brightness
            ))

    def _create_particle(self, pos, color, count, y_dir, life, speed_scale):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_scale
            vel = pygame.Vector2(math.cos(angle), math.sin(angle) if y_dir == 0 else y_dir) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(life // 2, life),
                'radius': self.np_random.uniform(1, 3),
                'color': color
            })
            
    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': color
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
            
        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            alpha = int(255 * (p['life'] / 30))
            color = p['color'] + (alpha,)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

        # Player projectiles
        for proj in self.player_projectiles:
            start = (int(proj.x), int(proj.y))
            end = (int(proj.x), int(proj.y + 8))
            pygame.draw.line(self.screen, self.COLOR_PROJECTILE, start, end, 2)
            
        # Alien projectiles
        for proj in self.alien_projectiles:
            pos = (int(proj.x), int(proj.y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 3, self.COLOR_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 3, self.COLOR_ALIEN)

        # Aliens
        for alien in self.aliens:
            pos = (int(alien['pos'].x), int(alien['pos'].y))
            pts = [
                (pos[0], pos[1] - 8), (pos[0] - 8, pos[1] + 4), 
                (pos[0] + 8, pos[1] + 4)
            ]
            pygame.gfxdraw.filled_trigon(self.screen, pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], self.COLOR_ALIEN_GLOW)
            pygame.gfxdraw.aatrigon(self.screen, pts[0][0], pts[0][1], pts[1][0], pts[1][1], pts[2][0], pts[2][1], self.COLOR_ALIEN)

        # Player
        if self.player_lives > 0:
            is_invulnerable = self.player_invulnerability_timer > 0
            if not (is_invulnerable and self.steps % 6 < 3):
                px, py = int(self.player_pos.x), int(self.player_pos.y)
                player_points = [(px, py - 10), (px - 8, py + 8), (px + 8, py + 8)]
                glow_points = [(px, py - 15), (px - 12, py + 12), (px + 12, py + 12)]
                
                pygame.gfxdraw.filled_trigon(self.screen, glow_points[0][0], glow_points[0][1], glow_points[1][0], glow_points[1][1], glow_points[2][0], glow_points[2][1], self.COLOR_PLAYER_GLOW)
                pygame.gfxdraw.filled_trigon(self.screen, player_points[0][0], player_points[0][1], player_points[1][0], player_points[1][1], player_points[2][0], player_points[2][1], self.COLOR_PLAYER)
                pygame.gfxdraw.aatrigon(self.screen, player_points[0][0], player_points[0][1], player_points[1][0], player_points[1][1], player_points[2][0], player_points[2][1], self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - 150, 10))
        for i in range(self.player_lives):
            px, py = self.WIDTH - 80 + i * 20, 18
            pts = [(px, py - 5), (px - 4, py + 4), (px + 4, py + 4)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, pts, 1)

        # Wave display
        if self.wave_clear_timer > 0:
            wave_text_str = f"WAVE {self.current_wave - 1} CLEARED" if self.current_wave > 1 else "GET READY"
            wave_text = self.font_wave.render(wave_text_str, True, self.COLOR_UI_TEXT)
            text_rect = wave_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(wave_text, text_rect)
        elif not self.game_over:
            wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
            text_rect = wave_text.get_rect(center=(self.WIDTH/2, self.HEIGHT - 20))
            self.screen.blit(wave_text, text_rect)
            
        # Game Over / Win message
        if self.game_over:
            msg = "MISSION COMPLETE" if self.current_wave > self.MAX_WAVES else "GAME OVER"
            end_text = self.font_wave.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It's a good way to test and debug your environment
    env = GameEnv()
    obs, info = env.reset()
    
    # Use a dummy window for rendering if you want to see the game
    pygame.display.set_caption("Arcade Racer Test")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    terminated = False
    total_reward = 0
    
    # Main game loop
    running = True
    while running:
        # Get player input from keyboard
        keys = pygame.key.get_pressed()
        
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0

        # Cap the frame rate
        env.clock.tick(env.FPS)

    env.close()