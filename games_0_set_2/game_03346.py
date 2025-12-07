
# Generated: 2025-08-27T23:05:24.804955
# Source Brief: brief_03346.md
# Brief Index: 3346

        
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
        "Controls: ←→ to move, ↑ to jump. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Control a jumping, shooting robot in a side-scrolling arena to defeat waves of enemy robots."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GROUND = (40, 45, 60)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GUN = (200, 200, 220)
        self.COLOR_ENEMY = (255, 80, 80)
        self.COLOR_ENEMY_EYE = (255, 255, 0)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 100)
        self.COLOR_PROJECTILE_ENEMY = (255, 150, 50)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR_BG = (80, 20, 20)
        self.COLOR_HEALTH_BAR = (200, 40, 40)

        # Game constants
        self.GROUND_Y = self.HEIGHT - 50
        self.GRAVITY = 0.6
        self.MAX_STEPS = 1000

        # Game state variables initialized in reset
        self.player = {}
        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self.steps = 0
        self.score = 0
        self.wave_number = 0
        self.game_over = False
        self.np_random = None

        self._generate_background()
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for final submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state for a new episode
        if options and options.get("full_reset", False):
            self.wave_number = 1
            self.score = 0
        else:
            # If not a full reset, and the previous wave was cleared, advance the wave
            if not self.game_over or (self.game_over and self.player['health'] > 0):
                 self.wave_number += 1
            else: # If player died, reset to wave 1
                 self.wave_number = 1
                 self.score = 0

        self.steps = 0
        self.game_over = False
        
        self.player = {
            'rect': pygame.Rect(self.WIDTH // 2 - 15, self.GROUND_Y - 30, 30, 30),
            'vx': 0, 'vy': 0,
            'health': 100, 'max_health': 100,
            'on_ground': True,
            'shoot_cooldown': 0,
            'last_move_dir': 1 # 1 for right, -1 for left
        }

        self.enemies = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Player Logic ---
        # Horizontal movement
        target_vx = 0
        if movement == 3:  # Left
            target_vx = -5
            self.player['last_move_dir'] = -1
        elif movement == 4:  # Right
            target_vx = 5
            self.player['last_move_dir'] = 1
        self.player['vx'] += (target_vx - self.player['vx']) * 0.3 # Smooth acceleration

        # Jumping
        if movement == 1 and self.player['on_ground']:
            self.player['vy'] = -12  # Jump strength
            self.player['on_ground'] = False
            # sfx: jump_sound()

        # Apply physics
        self.player['vy'] += self.GRAVITY
        self.player['rect'].x += int(self.player['vx'])
        self.player['rect'].y += int(self.player['vy'])
        
        # Collision with ground and screen boundaries
        self.player['rect'].left = max(0, self.player['rect'].left)
        self.player['rect'].right = min(self.WIDTH, self.player['rect'].right)
        if self.player['rect'].bottom >= self.GROUND_Y:
            self.player['rect'].bottom = self.GROUND_Y
            self.player['vy'] = 0
            self.player['on_ground'] = True
        
        # Shooting
        self.player['shoot_cooldown'] = max(0, self.player['shoot_cooldown'] - 1)
        if space_held and self.player['shoot_cooldown'] == 0:
            self.player['shoot_cooldown'] = 10 # 3 shots per second
            proj_x = self.player['rect'].centerx + 20 * self.player['last_move_dir']
            proj_y = self.player['rect'].centery - 5
            self.player_projectiles.append({
                'rect': pygame.Rect(proj_x, proj_y, 8, 4),
                'vx': 15 * self.player['last_move_dir']
            })
            # sfx: shoot_laser()
            self._create_particles(proj_x, proj_y, self.COLOR_PROJECTILE_PLAYER, 5, 2) # Muzzle flash

        # --- Enemy Logic ---
        for enemy in self.enemies:
            # Movement
            dx = self.player['rect'].centerx - enemy['rect'].centerx
            if abs(dx) > 50:
                enemy['rect'].x += 1 if dx > 0 else -1
            
            # Shooting
            enemy['shoot_cooldown'] = max(0, enemy['shoot_cooldown'] - 1)
            if enemy['shoot_cooldown'] == 0:
                enemy['shoot_cooldown'] = enemy['fire_rate']
                proj_x = enemy['rect'].centerx
                proj_y = enemy['rect'].centery
                self.enemy_projectiles.append({
                    'rect': pygame.Rect(proj_x, proj_y, 6, 6),
                    'vx': -6 if self.player['rect'].centerx < proj_x else 6
                })
                # sfx: enemy_shoot()

        # --- Projectile Logic ---
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj['rect'].x += proj['vx']
            if not self.screen.get_rect().colliderect(proj['rect']):
                self.player_projectiles.remove(proj)
                continue
            
            for enemy in self.enemies[:]:
                if proj['rect'].colliderect(enemy['rect']):
                    reward += 0.1
                    enemy['health'] -= 25
                    self._create_particles(proj['rect'].centerx, proj['rect'].centery, self.COLOR_ENEMY, 10, 3) # Hit spark
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    if enemy['health'] <= 0:
                        reward += 1.0
                        self.score += 100
                        self._create_particles(enemy['rect'].centerx, enemy['rect'].centery, (255,165,0), 40, 5) # Explosion
                        # sfx: explosion_sound()
                        self.enemies.remove(enemy)
                    break

        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj['rect'].x += proj['vx']
            if not self.screen.get_rect().colliderect(proj['rect']):
                self.enemy_projectiles.remove(proj)
                continue
            
            if proj['rect'].colliderect(self.player['rect']):
                reward -= 0.1
                self.player['health'] -= 10
                self._create_particles(proj['rect'].centerx, proj['rect'].centery, self.COLOR_PLAYER, 15, 3) # Player hit spark
                if proj in self.enemy_projectiles: self.enemy_projectiles.remove(proj)

        # --- Particle Logic ---
        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Termination Checks ---
        self.steps += 1
        terminated = False
        
        if self.player['health'] <= 0:
            self.player['health'] = 0
            terminated = True
            self.game_over = True
            reward = -100.0
            # sfx: player_death_sound()

        if not self.enemies:
            terminated = True
            self.game_over = True
            reward += 10.0 * self.wave_number
            # sfx: wave_clear_sound()

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        # Score bonus for surviving longer
        if not terminated:
            self.score += 1

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_wave(self):
        num_enemies = 10
        base_fire_rate = 120 # 2 seconds at 60fps logic rate
        fire_rate_scaling = 3 # gets faster by 0.05s per wave
        
        for i in range(num_enemies):
            x = self.np_random.integers(0, self.WIDTH)
            y = self.GROUND_Y - 25
            self.enemies.append({
                'rect': pygame.Rect(x, y, 25, 25),
                'health': 100,
                'shoot_cooldown': self.np_random.integers(0, base_fire_rate),
                'fire_rate': max(30, base_fire_rate - (self.wave_number - 1) * fire_rate_scaling)
            })

    def _create_particles(self, x, y, color, count, max_speed):
        for _ in range(count):
            self.particles.append({
                'x': x, 'y': y,
                'vx': self.np_random.uniform(-max_speed, max_speed),
                'vy': self.np_random.uniform(-max_speed, max_speed),
                'life': self.np_random.integers(10, 20),
                'color': color
            })
    
    def _generate_background(self):
        self.background_surface = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.background_surface.fill(self.COLOR_BG)
        # Distant buildings
        for i in range(50):
            x = random.randint(0, self.WIDTH)
            w = random.randint(20, 80)
            h = random.randint(50, 200)
            color = tuple(c * 0.5 for c in self.COLOR_GROUND)
            pygame.draw.rect(self.background_surface, color, (x, self.GROUND_Y - h, w, h))
        # Closer buildings
        for i in range(20):
            x = random.randint(0, self.WIDTH)
            w = random.randint(30, 100)
            h = random.randint(100, 250)
            color = tuple(c * 0.7 for c in self.COLOR_GROUND)
            pygame.draw.rect(self.background_surface, color, (x, self.GROUND_Y - h, w, h))


    def _get_observation(self):
        # Blit pre-rendered background
        self.screen.blit(self.background_surface, (0, 0))
        
        # Draw ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))
        
        # Render game elements
        self._render_particles()
        self._render_projectiles()
        self._render_enemies()
        self._render_player()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_player(self):
        # Body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player['rect'], border_radius=4)
        # Gun
        gun_x = self.player['rect'].centerx + (10 * self.player['last_move_dir'])
        gun_y = self.player['rect'].centery - 5
        gun_rect = pygame.Rect(0, 0, 15, 8)
        if self.player['last_move_dir'] > 0:
            gun_rect.midleft = (self.player['rect'].centerx, gun_y)
        else:
            gun_rect.midright = (self.player['rect'].centerx, gun_y)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GUN, gun_rect, border_radius=2)

    def _render_enemies(self):
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy['rect'], border_radius=3)
            eye_x = enemy['rect'].centerx + (5 if self.player['rect'].centerx > enemy['rect'].centerx else -5)
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_EYE, (eye_x, enemy['rect'].centery - 5), 3)

    def _render_projectiles(self):
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_PLAYER, proj['rect'], border_radius=2)
        for proj in self.enemy_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_PROJECTILE_ENEMY, proj['rect'].center, proj['rect'].width)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, 255 * (p['life'] / 20.0))
            size = max(1, int(p['life'] / 4))
            color_with_alpha = (*p['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), size, color_with_alpha)

    def _render_ui(self):
        # Score and Wave
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}", True, self.COLOR_UI_TEXT)
        wave_text_rect = wave_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(wave_text, wave_text_rect)

        # Player Health Bar
        health_bar_width = 200
        health_bar_height = 20
        health_bar_x = (self.WIDTH - health_bar_width) // 2
        health_bar_y = 10
        health_pct = self.player['health'] / self.player['max_health']
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (health_bar_x, health_bar_y, health_bar_width, health_bar_height), border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (health_bar_x, health_bar_y, int(health_bar_width * health_pct), health_bar_height), border_radius=4)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "player_health": self.player['health'],
            "enemies_left": len(self.enemies)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To display the game, we need a different setup
    pygame.display.set_caption("Robot Arena")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard keys to actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    running = True
    while running:
        # --- Human Controls ---
        movement_action = 0  # No-op
        space_action = 0
        shift_action = 0

        keys = pygame.key.get_pressed()
        for key, action_val in key_map.items():
            if keys[key]:
                movement_action = action_val
                break # Prioritize first key found in map order
        
        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        # We just need to convert it back to a Pygame surface to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}, Wave: {info['wave']}")
            obs, info = env.reset()

        env.clock.tick(30) # Control the frame rate for human play

    env.close()