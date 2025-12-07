
# Generated: 2025-08-27T20:00:03.206179
# Source Brief: brief_02316.md
# Brief Index: 2316

        
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
        "Controls: ↑↓ to aim cannon. Hold space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend Earth from descending alien hordes in this side-view, procedurally generated arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Game constants
        self.MAX_STEPS = 1000
        self.MAX_WAVES = 10
        self.GROUND_Y = self.HEIGHT - 40
        self.CANNON_ROTATION_SPEED = 2.5
        self.PROJECTILE_SPEED = 15
        self.FIRE_COOLDOWN_FRAMES = 8
        self.BASE_ALIEN_SPEED = 1.0

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_GROUND = (60, 40, 20)
        self.COLOR_CANNON = (100, 255, 100)
        self.COLOR_PROJECTILE = (150, 255, 150)
        self.COLOR_ALIEN = (255, 80, 80)
        self.COLOR_EXPLOSION = [(255, 255, 100), (255, 150, 50)]
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_GAMEOVER = (255, 50, 50)
        self.COLOR_WIN = (50, 255, 50)

        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)

        # Game state variables
        self.cannon_angle = 90.0
        self.projectiles = []
        self.aliens = []
        self.particles = []
        self.stars = []
        self.current_wave = 1
        self.alien_speed = 1.0
        self.aliens_in_wave_count = 0
        self.fire_cooldown = 0
        self.win = False

        self._generate_stars()
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.cannon_angle = 90.0
        self.projectiles = []
        self.aliens = []
        self.particles = []
        
        self.current_wave = 1
        self.fire_cooldown = 0

        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.1  # Survival reward

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Handle player actions
        if movement == 1:  # Rotate up (counter-clockwise)
            self.cannon_angle -= self.CANNON_ROTATION_SPEED
        elif movement == 2:  # Rotate down (clockwise)
            self.cannon_angle += self.CANNON_ROTATION_SPEED
        self.cannon_angle = np.clip(self.cannon_angle, 15, 165)

        if space_held and self.fire_cooldown == 0:
            self._fire_projectile()
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES
            # sfx: player_shoot.wav

        # Update game logic
        self.steps += 1
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        miss_penalty = self._update_projectiles()
        reward += miss_penalty

        self._update_aliens()
        self._update_particles()
        
        hit_reward = self._handle_collisions()
        reward += hit_reward

        # Check for wave completion
        if not self.aliens and not self.game_over:
            reward += 100
            self.current_wave += 1
            if self.current_wave > self.MAX_WAVES:
                self.win = True
                self.game_over = True
            else:
                self._spawn_wave()
                # sfx: wave_complete.wav

        # Check termination conditions
        terminated = self.game_over
        for alien in self.aliens:
            if alien['rect'].bottom >= self.GROUND_Y:
                terminated = True
                reward = -100
                # sfx: game_over.wav
                break
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
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
            "aliens_remaining": len(self.aliens)
        }

    # --- Helper Methods ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, self.HEIGHT)
            size = random.randint(1, 2)
            self.stars.append(((x, y), size))

    def _spawn_wave(self):
        self.aliens = []
        num_aliens = 5 + self.current_wave
        self.aliens_in_wave_count = num_aliens
        self.alien_speed = self.BASE_ALIEN_SPEED + (self.current_wave - 1) * 0.2

        for _ in range(num_aliens):
            size = random.randint(20, 30)
            x = random.uniform(size, self.WIDTH - size)
            y = random.uniform(-250 * (self.current_wave / self.MAX_WAVES), -size)
            
            alien = {
                'rect': pygame.Rect(x, y, size, size),
                'shape': random.choice(['square', 'triangle', 'pentagon']),
                'h_speed': random.uniform(0.5, 1.5),
                'h_phase': random.uniform(0, 2 * math.pi),
                'h_amp': random.uniform(20, 80),
                'initial_x': x
            }
            self.aliens.append(alien)

    def _fire_projectile(self):
        angle_rad = math.radians(self.cannon_angle)
        cannon_tip_x = self.WIDTH // 2 + 50 * math.cos(angle_rad - math.pi / 2)
        cannon_tip_y = self.GROUND_Y - 10 - 50 * math.sin(angle_rad - math.pi / 2)
        
        vx = self.PROJECTILE_SPEED * math.cos(angle_rad - math.pi / 2)
        vy = -self.PROJECTILE_SPEED * math.sin(angle_rad - math.pi / 2)
        
        self.projectiles.append({
            'pos': [cannon_tip_x, cannon_tip_y],
            'vel': [vx, vy],
            'hit': False
        })
        # Muzzle flash
        for _ in range(5):
            self._create_particle(
                (cannon_tip_x, cannon_tip_y),
                life=random.randint(3, 7),
                speed_mult=0.5,
                radius=random.randint(2, 4),
                color=self.COLOR_EXPLOSION[0]
            )

    def _update_projectiles(self):
        miss_penalty = 0
        projectiles_to_keep = []
        for p in self.projectiles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            
            on_screen = 0 <= p['pos'][0] < self.WIDTH and 0 <= p['pos'][1] < self.HEIGHT
            if on_screen:
                projectiles_to_keep.append(p)
            elif not p['hit']:
                miss_penalty -= 0.2
        self.projectiles = projectiles_to_keep
        return miss_penalty

    def _update_aliens(self):
        for alien in self.aliens:
            alien['rect'].y += self.alien_speed
            time_factor = (self.steps + alien['h_phase']) * 0.05 * alien['h_speed']
            horizontal_offset = alien['h_amp'] * math.sin(time_factor)
            alien['rect'].x = alien['initial_x'] + horizontal_offset
            alien['rect'].x = np.clip(alien['rect'].x, 0, self.WIDTH - alien['rect'].width)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['radius'] -= 0.1

    def _handle_collisions(self):
        reward = 0
        aliens_to_remove = set()
        projectiles_to_remove = set()

        for i, proj in enumerate(self.projectiles):
            proj_rect = pygame.Rect(proj['pos'][0] - 4, proj['pos'][1] - 4, 8, 8)
            for j, alien in enumerate(self.aliens):
                if j in aliens_to_remove:
                    continue
                if proj_rect.colliderect(alien['rect']):
                    aliens_to_remove.add(j)
                    projectiles_to_remove.add(i)
                    proj['hit'] = True
                    
                    reward += 1.0  # Base hit reward
                    headshot_threshold = alien['rect'].top + alien['rect'].height / 3
                    if proj_rect.centery < headshot_threshold:
                        reward += 2.0  # Headshot bonus
                    
                    self._create_explosion(alien['rect'].center)
                    # sfx: explosion.wav
                    break
        
        self.aliens = [a for i, a in enumerate(self.aliens) if i not in aliens_to_remove]
        # Mark projectiles for removal but let _update_projectiles handle it
        # to correctly process miss penalties. This logic is simplified by setting 'hit' flag.
        
        return reward

    def _create_explosion(self, pos):
        for _ in range(20):
            self._create_particle(
                pos,
                life=random.randint(15, 30),
                speed_mult=random.uniform(1, 4),
                radius=random.randint(3, 6),
                color=random.choice(self.COLOR_EXPLOSION)
            )

    def _create_particle(self, pos, life, speed_mult, radius, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 1.5) * speed_mult
        vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.particles.append({
            'pos': list(pos),
            'vel': vel,
            'life': life,
            'radius': radius,
            'color': color
        })
        
    # --- Rendering Methods ---

    def _render_game(self):
        # Stars
        for pos, size in self.stars:
            color_val = 100 if size == 1 else 150
            self.screen.set_at(pos, (color_val, color_val, color_val))

        # Ground
        pygame.draw.rect(self.screen, self.COLOR_GROUND, (0, self.GROUND_Y, self.WIDTH, self.HEIGHT - self.GROUND_Y))

        # Cannon
        cannon_base_rect = pygame.Rect(self.WIDTH // 2 - 20, self.GROUND_Y - 20, 40, 20)
        pygame.draw.rect(self.screen, self.COLOR_CANNON, cannon_base_rect)
        pygame.draw.circle(self.screen, self.COLOR_CANNON, (self.WIDTH // 2, self.GROUND_Y - 10), 15)
        
        angle_rad = math.radians(self.cannon_angle)
        end_x = self.WIDTH // 2 + 50 * math.cos(angle_rad - math.pi/2)
        end_y = self.GROUND_Y - 10 - 50 * math.sin(angle_rad - math.pi/2)
        pygame.draw.line(self.screen, self.COLOR_CANNON, (self.WIDTH // 2, self.GROUND_Y - 10), (end_x, end_y), 8)

        # Projectiles
        for p in self.projectiles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 4, self.COLOR_PROJECTILE)

        # Aliens
        for alien in self.aliens:
            if alien['shape'] == 'square':
                pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'])
            elif alien['shape'] == 'triangle':
                p1 = alien['rect'].midtop
                p2 = alien['rect'].bottomleft
                p3 = alien['rect'].bottomright
                pygame.draw.polygon(self.screen, self.COLOR_ALIEN, [p1, p2, p3])
            elif alien['shape'] == 'pentagon':
                points = []
                for i in range(5):
                    angle = (i / 5.0) * 2 * math.pi - math.pi / 2
                    x = alien['rect'].centerx + alien['rect'].width/2 * math.cos(angle)
                    y = alien['rect'].centery + alien['rect'].height/2 * math.sin(angle)
                    points.append((x, y))
                pygame.draw.polygon(self.screen, self.COLOR_ALIEN, points)

        # Particles
        for p in self.particles:
            if p['radius'] > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                alpha = int(255 * (p['life'] / 30))
                color = (*p['color'], alpha)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_main.render(f"Wave: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Aliens remaining
        aliens_text = self.font_main.render(f"Aliens: {len(self.aliens)}", True, self.COLOR_TEXT)
        self.screen.blit(aliens_text, (self.WIDTH - aliens_text.get_width() - 10, self.HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = self.COLOR_WIN
            else:
                msg = "GAME OVER"
                color = self.COLOR_GAMEOVER
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Human controls
        keys = pygame.key.get_pressed()
        action.fill(0) # Reset action
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # 30 FPS

    pygame.quit()