
# Generated: 2025-08-27T20:28:10.403611
# Source Brief: brief_02470.md
# Brief Index: 2470

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A visually stunning side-view space shooter. Eliminate waves of descending invaders while dodging their projectiles and collecting power-ups."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_WAVES = 10
        self.MAX_STEPS = 10000

        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (50, 200, 255)
        self.COLOR_PLAYER_GLOW = (50, 200, 255, 50)
        self.COLOR_ENEMY_PROJECTILE = (255, 50, 50)
        self.COLOR_PLAYER_PROJECTILE = (100, 220, 255)
        self.COLOR_POWERUP_SHIELD = (255, 220, 50)
        self.COLOR_POWERUP_RAPID = (200, 50, 255)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.EXPLOSION_COLORS = [(255, 200, 50), (255, 100, 0), (200, 50, 0)]
        
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
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 40, bold=True)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 0
        self.player_pos = [0, 0]
        self.player_lives = 0
        self.player_fire_cooldown = 0
        self.player_projectiles = []
        self.invaders = []
        self.enemy_projectiles = []
        self.particles = []
        self.powerups = []
        self.stars = []
        self.invader_direction = 1
        self.invader_base_speed = 0
        self.invader_fire_rate = 0
        self.shield_timer = 0
        self.rapid_fire_timer = 0
        self.player_hit_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave = 1
        
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_lives = 3
        self.player_fire_cooldown = 0
        self.player_hit_timer = 0
        
        self.shield_timer = 0
        self.rapid_fire_timer = 0
        
        self.player_projectiles.clear()
        self.invaders.clear()
        self.enemy_projectiles.clear()
        self.particles.clear()
        self.powerups.clear()

        self.invader_direction = 1
        self._spawn_wave()

        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 3))
            for _ in range(150)
        ]

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.01  # Small survival reward per frame
        self.steps += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self._handle_input(action)
        self._update_game_state()
        
        collision_rewards = self._handle_collisions()
        reward += collision_rewards
        
        state_change_rewards = self._update_wave_state()
        reward += state_change_rewards

        self.game_over = self._check_termination()
        if self.game_over and self.player_lives <= 0:
            reward = -100.0
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # Player Movement
        if movement == 3:  # Left
            self.player_pos[0] -= 5
        elif movement == 4:  # Right
            self.player_pos[0] += 5
        self.player_pos[0] = np.clip(self.player_pos[0], 20, self.WIDTH - 20)

        # Player Firing
        cooldown_max = 5 if self.rapid_fire_timer > 0 else 15
        if space_held and self.player_fire_cooldown <= 0:
            # Sfx: Player shoot
            self.player_projectiles.append(pygame.Rect(self.player_pos[0] - 2, self.player_pos[1] - 20, 4, 15))
            self.player_fire_cooldown = cooldown_max

    def _update_game_state(self):
        # Cooldowns
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.shield_timer > 0: self.shield_timer -= 1
        if self.rapid_fire_timer > 0: self.rapid_fire_timer -= 1
        if self.player_hit_timer > 0: self.player_hit_timer -= 1
        
        # Player Projectiles
        self.player_projectiles = [p for p in self.player_projectiles if p.y > 0]
        for p in self.player_projectiles:
            p.y -= 10

        # Enemy Projectiles
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p.y < self.HEIGHT]
        for p in self.enemy_projectiles:
            p.y += 5

        # Powerups
        self.powerups = [pw for pw in self.powerups if pw['pos'][1] < self.HEIGHT]
        for pw in self.powerups:
            pw['pos'][1] += 1
            pw['rect'].center = tuple(pw['pos'])

        # Invaders
        self._update_invaders()
        
        # Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

        # Starfield
        for i, (x, y, speed) in enumerate(self.stars):
            y_new = (y + speed) % self.HEIGHT
            self.stars[i] = (x, y_new, speed)

    def _update_invaders(self):
        if not self.invaders:
            return

        move_down = False
        for invader in self.invaders:
            if (invader['rect'].right > self.WIDTH and self.invader_direction > 0) or \
               (invader['rect'].left < 0 and self.invader_direction < 0):
                move_down = True
                break
        
        if move_down:
            self.invader_direction *= -1
            for invader in self.invaders:
                invader['pos'][1] += 10
        
        for invader in self.invaders:
            invader['pos'][0] += self.invader_direction * self.invader_base_speed
            invader['rect'].center = (int(invader['pos'][0]), int(invader['pos'][1]))
            
            if random.random() < self.invader_fire_rate:
                # Sfx: Enemy shoot
                self.enemy_projectiles.append(pygame.Rect(invader['rect'].centerx - 2, invader['rect'].centery, 4, 10))

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Invaders
        projectiles_to_remove = []
        invaders_to_remove = []
        for p_idx, p in enumerate(self.player_projectiles):
            for i_idx, invader in enumerate(self.invaders):
                if i_idx in invaders_to_remove: continue
                if p.colliderect(invader['rect']):
                    projectiles_to_remove.append(p_idx)
                    invaders_to_remove.append(i_idx)
                    
                    # Sfx: Explosion
                    self._create_explosion(invader['rect'].center)
                    self.score += 10
                    reward += 1.0

                    if random.random() < 0.2: # 20% powerup drop chance
                        self._spawn_powerup(invader['rect'].center)
                    break

        self.player_projectiles = [p for i, p in enumerate(self.player_projectiles) if i not in projectiles_to_remove]
        self.invaders = [inv for i, inv in enumerate(self.invaders) if i not in invaders_to_remove]

        # Enemy projectiles vs Player
        player_rect = pygame.Rect(self.player_pos[0] - 18, self.player_pos[1] - 10, 36, 20)
        projectiles_to_remove.clear()
        if self.player_hit_timer <= 0:
            for p_idx, p in enumerate(self.enemy_projectiles):
                if p.colliderect(player_rect):
                    projectiles_to_remove.append(p_idx)
                    if self.shield_timer > 0:
                        # Sfx: Shield hit
                        self.shield_timer = 0
                    else:
                        # Sfx: Player hit
                        self.player_lives -= 1
                        self.player_hit_timer = 60 # 2 seconds of invincibility
                        self._create_explosion(self.player_pos, count=15, is_player=True)
                    break
        self.enemy_projectiles = [p for i, p in enumerate(self.enemy_projectiles) if i not in projectiles_to_remove]
        
        # Player vs Powerups
        powerups_to_remove = []
        for pw_idx, pw in enumerate(self.powerups):
            if player_rect.colliderect(pw['rect']):
                powerups_to_remove.append(pw_idx)
                # Sfx: Powerup collect
                self.score += 50
                reward += 5.0
                if pw['type'] == 'shield':
                    self.shield_timer = 300 # 10 seconds
                elif pw['type'] == 'rapid_fire':
                    self.rapid_fire_timer = 300 # 10 seconds
        self.powerups = [pw for i, pw in enumerate(self.powerups) if i not in powerups_to_remove]
        
        return reward

    def _update_wave_state(self):
        reward = 0
        if not self.invaders and not self.game_over:
            self.wave += 1
            if self.wave > self.MAX_WAVES:
                # Sfx: Game win
                self.game_over = True
                reward += 100.0 # Win game reward
                self.score += 5000
            else:
                # Sfx: Wave complete
                self.score += 1000
                reward += 100.0
                self._spawn_wave()
        return reward

    def _check_termination(self):
        if self.player_lives <= 0:
            # Sfx: Game over
            return True
        for invader in self.invaders:
            if invader['rect'].bottom >= self.HEIGHT - 20:
                # Sfx: Invader reached bottom
                self.player_lives = 0 # Instant loss
                return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_powerups()
        self._render_invaders()
        self._render_player()
        self._render_projectiles()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "lives": self.player_lives,
        }

    def _spawn_wave(self):
        self.invaders.clear()
        self.invader_base_speed = 0.5 + self.wave * 0.2
        self.invader_fire_rate = 0.001 + self.wave * 0.0005
        
        rows, cols = 5, 10
        x_spacing, y_spacing = 50, 40
        start_x = (self.WIDTH - (cols - 1) * x_spacing) / 2
        start_y = 60
        
        for r in range(rows):
            for c in range(cols):
                pos = [start_x + c * x_spacing, start_y + r * y_spacing]
                rect = pygame.Rect(0, 0, 30, 20)
                rect.center = tuple(pos)
                self.invaders.append({'pos': pos, 'rect': rect, 'type': r % 3})

    def _spawn_powerup(self, pos):
        ptype = random.choice(['shield', 'rapid_fire'])
        rect = pygame.Rect(0, 0, 20, 20)
        rect.center = pos
        self.powerups.append({'pos': list(pos), 'rect': rect, 'type': ptype})

    def _create_explosion(self, pos, count=30, is_player=False):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            color = random.choice(self.EXPLOSION_COLORS)
            life = random.randint(15, 40)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})

    # --- RENDER METHODS ---

    def _render_background(self):
        for x, y, speed in self.stars:
            size = speed
            color_val = 50 + speed * 40
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (x, y, size, size))

    def _render_player(self):
        x, y = int(self.player_pos[0]), int(self.player_pos[1])
        
        # Make player flash when hit
        if (self.player_hit_timer // 4) % 2 == 1:
            return

        # Player ship polygon
        points = [(x, y - 15), (x - 18, y + 10), (x + 18, y + 10)]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

        # Engine glow
        engine_y = y + 12
        pygame.draw.circle(self.screen, (255, 255, 255), (x, engine_y), 5)
        pygame.gfxdraw.filled_circle(self.screen, x, engine_y, 5, (255, 150, 50))
        
        # Shield effect
        if self.shield_timer > 0:
            alpha = min(255, 50 + (self.shield_timer % 15) * 10)
            pygame.gfxdraw.aacircle(self.screen, x, y, 30, (*self.COLOR_POWERUP_SHIELD, alpha))
            pygame.gfxdraw.aacircle(self.screen, x, y, 31, (*self.COLOR_POWERUP_SHIELD, alpha//2))

    def _render_invaders(self):
        for invader in self.invaders:
            itype = invader['type']
            rect = invader['rect']
            if itype == 0:
                color = (255, 80, 80)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)
            elif itype == 1:
                color = (80, 255, 80)
                points = [rect.midtop, rect.bottomleft, rect.bottomright]
                pygame.draw.polygon(self.screen, color, points)
            else:
                color = (80, 80, 255)
                pygame.draw.ellipse(self.screen, color, rect)
            
            # Simple glow
            glow_color = (*color, 60)
            glow_rect = rect.inflate(8, 8)
            pygame.draw.rect(self.screen, glow_color, glow_rect, border_radius=8)

    def _render_projectiles(self):
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, p, border_radius=2)
            pygame.gfxdraw.box(self.screen, p.inflate(4, 4), (*self.COLOR_PLAYER_PROJECTILE, 50))
        for p in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJECTILE, p, border_radius=2)
            pygame.gfxdraw.box(self.screen, p.inflate(4, 4), (*self.COLOR_ENEMY_PROJECTILE, 50))

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40.0))
            color = (*p['color'], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            pygame.gfxdraw.pixel(self.screen, *pos, color)
            pygame.gfxdraw.pixel(self.screen, pos[0]+1, pos[1], color)
            pygame.gfxdraw.pixel(self.screen, pos[0], pos[1]+1, color)
            pygame.gfxdraw.pixel(self.screen, pos[0]+1, pos[1]+1, color)

    def _render_powerups(self):
        for pw in self.powerups:
            if pw['type'] == 'shield':
                color = self.COLOR_POWERUP_SHIELD
                text = "S"
            else:
                color = self.COLOR_POWERUP_RAPID
                text = "R"
            
            pygame.draw.rect(self.screen, color, pw['rect'], border_radius=5)
            pygame.gfxdraw.box(self.screen, pw['rect'].inflate(6,6), (*color, 50))
            
            text_surf = self.font_small.render(text, True, self.COLOR_BG)
            text_rect = text_surf.get_rect(center=pw['rect'].center)
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives - 1):
            x, y = 25 + i * 30, self.HEIGHT - 25
            points = [(x, y - 8), (x - 10, y + 5), (x + 10, y + 5)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        if self.game_over:
            msg = "YOU WIN!" if self.wave > self.MAX_WAVES else "GAME OVER"
            color = (50, 255, 50) if self.wave > self.MAX_WAVES else (255, 50, 50)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Invader")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            obs, info = env.reset()
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS

    pygame.quit()