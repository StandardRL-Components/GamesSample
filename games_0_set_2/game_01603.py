import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move your ship. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Survive three waves of increasingly difficult alien attacks in this retro top-down arcade shooter."
    )

    # Frames auto-advance at 30fps.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.game_font = pygame.font.SysFont("dejavusansmono", 20)
        self.title_font = pygame.font.SysFont("dejavusansmono", 30, bold=True)

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 255)
        self.COLOR_PROJECTILE_ENEMY = (255, 100, 100)
        self.COLOR_WAVE_1 = (255, 50, 50)
        self.COLOR_WAVE_2 = (50, 150, 255)
        self.COLOR_WAVE_3 = (200, 50, 255)
        self.COLOR_EXPLOSION = [(255, 200, 0), (255, 100, 0), (255, 50, 0)]
        self.COLOR_UI = (200, 200, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (50, 50, 50)

        # Game parameters
        self.MAX_STEPS = 10000
        self.PLAYER_SPEED = 5
        self.PLAYER_MAX_HEALTH = 100
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PROJECTILE_SPEED = 10
        
        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []
        self.wave_number = 0
        self.wave_cleared_bonus_given = False
        self.reward_this_step = 0
        
        self.reset()
        
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.wave_cleared_bonus_given = False
        
        self.player = {
            'pos': pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 50),
            'health': self.PLAYER_MAX_HEALTH,
            'fire_cooldown': 0,
            'hit_timer': 0
        }
        
        self.enemies.clear()
        self.player_projectiles.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()

        # Create starfield
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'speed': self.np_random.uniform(0.5, 2.0),
                'radius': self.np_random.uniform(0.5, 1.5)
            })

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0.1 # Survival reward
        
        # Unpack action
        movement = action[0]
        space_held = action[1] == 1
        
        self._handle_input(movement, space_held)
        
        self._update_player()
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        self._update_stars()
        
        self._handle_collisions()

        self.steps += 1
        
        if len(self.enemies) == 0 and self.wave_number <= 3:
            if not self.wave_cleared_bonus_given:
                if self.wave_number == 3: # Game won
                    self.reward_this_step += 500
                    self.game_over = True
                else: # Wave cleared
                    self.reward_this_step += 100
                    self._spawn_wave()
                self.wave_cleared_bonus_given = True
        
        terminated = self._check_termination()
        reward = self.reward_this_step
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player['pos'].y -= self.PLAYER_SPEED
        if movement == 2: self.player['pos'].y += self.PLAYER_SPEED
        if movement == 3: self.player['pos'].x -= self.PLAYER_SPEED
        if movement == 4: self.player['pos'].x += self.PLAYER_SPEED
        
        # Firing
        if space_held and self.player['fire_cooldown'] <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append({
                'pos': self.player['pos'].copy(),
                'vel': pygame.Vector2(0, -self.PROJECTILE_SPEED),
            })
            self.player['fire_cooldown'] = self.PLAYER_FIRE_COOLDOWN

    def _update_player(self):
        # Cooldowns
        if self.player['fire_cooldown'] > 0:
            self.player['fire_cooldown'] -= 1
        if self.player['hit_timer'] > 0:
            self.player['hit_timer'] -= 1

        # Keep player in bounds
        self.player['pos'].x = np.clip(self.player['pos'].x, 20, self.WIDTH - 20)
        self.player['pos'].y = np.clip(self.player['pos'].y, 20, self.HEIGHT - 20)

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj['pos'] += proj['vel']
            if proj['pos'].y < 0:
                self.player_projectiles.remove(proj)
        
        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj['pos'] += proj['vel']
            if not (0 <= proj['pos'].x <= self.WIDTH and 0 <= proj['pos'].y <= self.HEIGHT):
                self.enemy_projectiles.remove(proj)

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement based on wave
            if self.wave_number == 1: # Horizontal sine wave
                enemy['pattern_state']['angle'] += 0.05
                enemy['pos'].x = enemy['pattern_state']['start_x'] + math.sin(enemy['pattern_state']['angle']) * 100
                enemy['pos'].y += enemy['speed'] * 0.5
            elif self.wave_number == 2: # Diagonal dive
                enemy['pos'] += enemy['pattern_state']['dir'] * enemy['speed']
            elif self.wave_number == 3: # Circling player
                target_pos = self.player['pos'] + pygame.Vector2(150, 0).rotate(enemy['pattern_state']['angle'])
                direction = (target_pos - enemy['pos']).normalize()
                enemy['pos'] += direction * enemy['speed']
                enemy['pattern_state']['angle'] += 1.5

            # Firing
            enemy['cooldown'] -= 1
            if enemy['cooldown'] <= 0:
                # sfx: enemy_shoot.wav
                if (self.player['pos'] - enemy['pos']).length() > 0: # Avoid normalizing zero vector
                    direction = (self.player['pos'] - enemy['pos']).normalize()
                    self.enemy_projectiles.append({
                        'pos': enemy['pos'].copy(),
                        'vel': direction * self.PROJECTILE_SPEED * 0.7
                    })
                enemy['cooldown'] = enemy['fire_rate']

            # Despawn if off-screen
            if enemy['pos'].y > self.HEIGHT + 20:
                self.enemies.remove(enemy)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] -= 0.1
            if p['lifespan'] <= 0 or p['radius'] <= 0:
                self.particles.remove(p)

    def _update_stars(self):
        for star in self.stars:
            star['pos'].y += star['speed']
            if star['pos'].y > self.HEIGHT:
                star['pos'].y = 0
                star['pos'].x = self.np_random.uniform(0, self.WIDTH)

    def _handle_collisions(self):
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                if proj['pos'].distance_to(enemy['pos']) < 15: # Collision
                    # sfx: explosion.wav
                    self._create_explosion(enemy['pos'], 20, self.COLOR_EXPLOSION)
                    self.enemies.remove(enemy)
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    
                    self.score += 1
                    self.reward_this_step += 1
                    break

        # Enemy projectiles vs player
        player_hitbox_radius = 10
        for proj in self.enemy_projectiles[:]:
            if proj['pos'].distance_to(self.player['pos']) < player_hitbox_radius and self.player['hit_timer'] <= 0:
                # sfx: player_hit.wav
                self.enemy_projectiles.remove(proj)
                self.player['health'] -= 10
                self.player['hit_timer'] = 30 # 1 sec invulnerability
                self.reward_this_step -= 0.2
                self._create_explosion(self.player['pos'], 10, [self.COLOR_PLAYER])
                if self.player['health'] <= 0:
                    self.game_over = True
                    self._create_explosion(self.player['pos'], 50, self.COLOR_EXPLOSION)
                break
    
    def _spawn_wave(self):
        self.wave_number += 1
        self.wave_cleared_bonus_given = False
        num_enemies = 5 + self.wave_number * 2
        base_speed = 1.0 + (self.wave_number - 1) * 0.5
        base_fire_rate = 60 - (self.wave_number - 1) * 6 # in frames (30fps)

        for i in range(num_enemies):
            if self.wave_number == 1:
                enemy = {
                    'pos': pygame.Vector2(self.WIDTH / 2 + (i - num_enemies/2) * 50, -20 - i * 30),
                    'color': self.COLOR_WAVE_1,
                    'speed': base_speed,
                    'fire_rate': base_fire_rate,
                    'cooldown': self.np_random.integers(0, base_fire_rate),
                    'pattern_state': {'angle': self.np_random.uniform(0, 2*math.pi), 'start_x': self.WIDTH / 2 + (i - num_enemies/2) * 50}
                }
            elif self.wave_number == 2:
                start_x = self.np_random.choice([0, self.WIDTH])
                direction = pygame.Vector2(self.WIDTH/2 - start_x, 100).normalize()
                enemy = {
                    'pos': pygame.Vector2(start_x, self.np_random.uniform(50, 150)),
                    'color': self.COLOR_WAVE_2,
                    'speed': base_speed,
                    'fire_rate': base_fire_rate,
                    'cooldown': self.np_random.integers(0, base_fire_rate),
                    'pattern_state': {'dir': direction}
                }
            elif self.wave_number == 3:
                angle = 360 / num_enemies * i
                enemy = {
                    'pos': pygame.Vector2(self.WIDTH / 2, -50) + pygame.Vector2(200, 0).rotate(angle),
                    'color': self.COLOR_WAVE_3,
                    'speed': base_speed,
                    'fire_rate': base_fire_rate,
                    'cooldown': self.np_random.integers(0, base_fire_rate),
                    'pattern_state': {'angle': angle}
                }
            else:
                return # No more waves
            self.enemies.append(enemy)

    def _create_explosion(self, position, num_particles, colors):
        for _ in range(num_particles):
            vel = pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3))
            self.particles.append({
                'pos': position.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'color': random.choice(colors),
                'lifespan': self.np_random.integers(15, 30)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.player['health'] <= 0:
            self.game_over = True
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if self.wave_number > 3 and len(self.enemies) == 0: # All 3 waves cleared
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
            "health": self.player['health'],
            "wave": self.wave_number,
        }

    def _render_game(self):
        # Stars
        for star in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 220), star['pos'], star['radius'])

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, p['radius']))
        
        # Player Projectiles
        for proj in self.player_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_PLAYER, proj['pos'], 3)

        # Enemy Projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_ENEMY, proj['pos'], 4)
        
        # Enemies
        for enemy in self.enemies:
            if self.wave_number == 1: # Square
                pygame.draw.rect(self.screen, enemy['color'], (enemy['pos'].x - 8, enemy['pos'].y - 8, 16, 16))
            elif self.wave_number == 2: # Diamond
                points = [(enemy['pos'].x, enemy['pos'].y - 10), (enemy['pos'].x + 10, enemy['pos'].y), 
                          (enemy['pos'].x, enemy['pos'].y + 10), (enemy['pos'].x - 10, enemy['pos'].y)]
                pygame.gfxdraw.aapolygon(self.screen, points, enemy['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, enemy['color'])
            elif self.wave_number == 3: # Triangle
                points = [(enemy['pos'].x, enemy['pos'].y - 10), (enemy['pos'].x + 10, enemy['pos'].y + 8),
                          (enemy['pos'].x - 10, enemy['pos'].y + 8)]
                pygame.gfxdraw.aapolygon(self.screen, points, enemy['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, enemy['color'])
        
        # Player
        if self.player['health'] > 0:
            # Invincibility flash
            if self.player['hit_timer'] > 0 and (self.steps // 3) % 2 == 0:
                return 

            p = self.player['pos']
            player_points = [(p.x, p.y - 15), (p.x + 10, p.y + 10), (p.x - 10, p.y + 10)]
            
            # Glow effect
            glow_surface = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.polygon(glow_surface, self.COLOR_PLAYER_GLOW, 
                                [(pt[0] - p.x + 20, pt[1] - p.y + 20) for pt in player_points])
            # Manual blur by scaling down and up, as a fallback for older Pygame versions
            glow_surface = pygame.transform.smoothscale(glow_surface, (glow_surface.get_width() // 4, glow_surface.get_height() // 4))
            glow_surface = pygame.transform.smoothscale(glow_surface, (40, 40))
            self.screen.blit(glow_surface, (p.x - 20, p.y - 20))

            # Main ship
            pygame.gfxdraw.aapolygon(self.screen, player_points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, player_points, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.game_font.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_str = f"WAVE: {self.wave_number}" if self.wave_number <= 3 else "VICTORY!"
        wave_text = self.game_font.render(wave_str, True, self.COLOR_UI)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Health Bar
        health_pct = max(0, self.player['health'] / self.PLAYER_MAX_HEALTH)
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = self.HEIGHT - bar_height - 10
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, bar_width * health_pct, bar_height))

        # Game Over / Win Message
        if self.game_over:
            msg = "GAME OVER" if self.player['health'] <= 0 else "YOU WIN!"
            msg_text = self.title_font.render(msg, True, self.COLOR_UI)
            msg_rect = msg_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(msg_text, msg_rect)

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
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
    env = GameEnv(render_mode="rgb_array")
    
    # Use a dummy window to display the game
    pygame.display.set_caption("Space Shooter")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Main game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard input
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
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Reset if the game is over
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        # Cap the frame rate
        env.clock.tick(30)

    env.close()