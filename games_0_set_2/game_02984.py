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
        "Controls: ←→ to move. Hold Space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A modern, procedurally generated take on Space Invaders. "
        "Survive 3 stages of alien waves to win. "
        "Aggressive play is rewarded."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and timing
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30

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
        self.COLOR_BG = (15, 15, 35)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 50, 100)
        self.COLOR_PROJECTILE_PLAYER = (200, 255, 255)
        self.COLOR_PROJECTILE_ALIEN = (255, 200, 0)
        self.COLOR_ALIEN_BASIC = (255, 80, 80)
        self.COLOR_ALIEN_FAST = (80, 255, 80)
        self.COLOR_ALIEN_TANK = (255, 150, 50)
        self.COLOR_TEXT = (220, 220, 220)
        self.EXPLOSION_COLORS = [(255, 255, 100), (255, 150, 50), (255, 50, 50)]

        # Game constants
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 5 # frames
        self.PLAYER_SIZE = (20, 15)
        self.PROJECTILE_SPEED = 12

        self.MAX_STAGES = 3
        self.STAGE_TIME = 60 * self.FPS

        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_fire_timer = None
        self.aliens = None
        self.projectiles = None
        self.particles = None
        self.stars = None
        self.score = None
        self.stage = None
        self.stage_timer = None
        self.game_over = None
        self.game_won = None
        self.alien_direction = None
        self.alien_descent_speed = None
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Player state
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = 3
        self.player_fire_timer = 0

        # Game state
        self.aliens = []
        self.projectiles = []
        self.particles = []
        self.score = 0
        self.steps = 0
        self.stage = 1
        self.game_over = False
        self.game_won = False
        
        # Setup background stars
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'speed': self.np_random.uniform(0.2, 1.2),
                'radius': self.np_random.uniform(0.5, 1.5)
            })
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        self.aliens.clear()
        self.projectiles.clear()
        self.stage_timer = self.STAGE_TIME
        self.alien_direction = 1
        self.alien_descent_speed = 20

        stage_config = self._get_stage_config(self.stage)
        rows = stage_config['rows']
        cols = stage_config['cols']
        alien_types = stage_config['types']
        
        grid_width = cols * 40
        start_x = (self.WIDTH - grid_width) / 2
        start_y = 50

        for r in range(rows):
            for c in range(cols):
                alien_type_key = alien_types[r % len(alien_types)]
                alien_info = self._get_alien_info(alien_type_key)
                
                self.aliens.append({
                    'pos': pygame.Vector2(start_x + c * 40, start_y + r * 40),
                    'type': alien_type_key,
                    'hp': alien_info['hp'],
                    'score': alien_info['score'],
                    'size': alien_info['size'],
                    'fire_cooldown': self.np_random.integers(0, 100),
                })
    
    def _get_stage_config(self, stage):
        if stage == 1:
            return {'rows': 3, 'cols': 8, 'types': ['basic']}
        elif stage == 2:
            return {'rows': 4, 'cols': 9, 'types': ['basic', 'fast']}
        else: # Stage 3
            return {'rows': 5, 'cols': 10, 'types': ['basic', 'tank', 'fast']}

    def _get_alien_info(self, alien_type):
        if alien_type == 'basic':
            return {'hp': 1, 'score': 1, 'size': (24, 18), 'color': self.COLOR_ALIEN_BASIC}
        elif alien_type == 'fast':
            return {'hp': 1, 'score': 2, 'size': (20, 15), 'color': self.COLOR_ALIEN_FAST}
        elif alien_type == 'tank':
            return {'hp': 3, 'score': 3, 'size': (30, 22), 'color': self.COLOR_ALIEN_TANK}
        return {}
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        reward = -0.01  # Time penalty

        # 1. Handle Input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if movement in [3, 4]:
            reward += 0.01 # Reward for moving
        
        self._handle_input(movement, space_held)

        # 2. Update Game State
        self._update_player()
        update_reward = self._update_aliens()
        reward += update_reward
        self._update_projectiles()
        self._update_particles()
        self._update_timers()
        
        # 3. Handle Collisions
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # 4. Check for stage/game progression
        stage_reward = self._check_progression()
        reward += stage_reward

        # Add reward from this step's actions
        reward += self.reward_this_step

        # 5. Check Termination Conditions
        terminated = self._check_termination()
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held):
        # Horizontal movement
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Firing
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            self.projectiles.append({
                'pos': pygame.Vector2(self.player_pos.x, self.player_pos.y - self.PLAYER_SIZE[1]),
                'vel': pygame.Vector2(0, -self.PROJECTILE_SPEED),
                'owner': 'player',
                'color': self.COLOR_PROJECTILE_PLAYER,
                'size': (4, 12)
            })
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN
            
            # Reward for shooting near an alien
            for alien in self.aliens:
                if abs(alien['pos'].x - self.player_pos.x) < 50:
                    self.reward_this_step += 0.05
                    break # Only once per shot

    def _update_player(self):
        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_SIZE[0]/2, self.WIDTH - self.PLAYER_SIZE[0]/2)
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
    
    def _update_aliens(self):
        if not self.aliens:
            return 0
        
        stage_speed_mult = 1 + (self.stage - 1) * 0.2
        stage_fire_rate_mult = 1 + (self.stage - 1) * 0.5
        
        move_sideways = True
        move_down = False

        for alien in self.aliens:
            if (alien['pos'].x <= alien['size'][0]/2 and self.alien_direction == -1) or \
               (alien['pos'].x >= self.WIDTH - alien['size'][0]/2 and self.alien_direction == 1):
                move_down = True
                self.alien_direction *= -1
                break

        for alien in self.aliens:
            if move_down:
                alien['pos'].y += self.alien_descent_speed
            else:
                base_speed = 1.0 if alien['type'] != 'fast' else 1.5
                alien['pos'].x += self.alien_direction * base_speed * stage_speed_mult

            # Alien firing logic
            fire_chance = 0.001 * stage_fire_rate_mult
            if self.np_random.random() < fire_chance:
                # sfx: alien_shoot.wav
                self.projectiles.append({
                    'pos': pygame.Vector2(alien['pos'].x, alien['pos'].y + alien['size'][1]),
                    'vel': pygame.Vector2(0, self.PROJECTILE_SPEED * 0.5),
                    'owner': 'alien',
                    'color': self.COLOR_PROJECTILE_ALIEN,
                    'size': (4, 8)
                })
        return 0

    def _update_projectiles(self):
        for proj in self.projectiles[:]:
            proj['pos'] += proj['vel']
            if not (0 < proj['pos'].y < self.HEIGHT):
                self.projectiles.remove(proj)

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
    
    def _update_timers(self):
        self.stage_timer -= 1
        if self.stage_timer <= 0:
            self.game_over = True
    
    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE[0]/2, self.player_pos.y - self.PLAYER_SIZE[1]/2, *self.PLAYER_SIZE)

        # Player projectiles vs Aliens
        for proj in self.projectiles[:]:
            if proj['owner'] == 'player':
                proj_rect = pygame.Rect(proj['pos'].x - proj['size'][0]/2, proj['pos'].y - proj['size'][1]/2, *proj['size'])
                for alien in self.aliens[:]:
                    alien_rect = pygame.Rect(alien['pos'].x - alien['size'][0]/2, alien['pos'].y - alien['size'][1]/2, *alien['size'])
                    if proj_rect.colliderect(alien_rect):
                        # sfx: hit.wav
                        self.projectiles.remove(proj)
                        alien['hp'] -= 1
                        if alien['hp'] <= 0:
                            # sfx: explosion.wav
                            self._create_explosion(alien['pos'], [self._get_alien_info(alien['type'])['color']])
                            reward += alien['score']
                            self.score += alien['score']
                            self.aliens.remove(alien)
                        break
        
        # Alien projectiles vs Player
        for proj in self.projectiles[:]:
            if proj['owner'] == 'alien':
                proj_rect = pygame.Rect(proj['pos'].x - proj['size'][0]/2, proj['pos'].y - proj['size'][1]/2, *proj['size'])
                if player_rect.colliderect(proj_rect):
                    # sfx: player_hit.wav
                    self.projectiles.remove(proj)
                    self.player_lives -= 1
                    reward -= 1
                    self._create_explosion(self.player_pos, [self.COLOR_PLAYER, (255,255,255)])
                    if self.player_lives > 0:
                        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40) # Reset position
                    else:
                        self.game_over = True
                    break
        
        # Aliens vs Player
        for alien in self.aliens:
            alien_rect = pygame.Rect(alien['pos'].x - alien['size'][0]/2, alien['pos'].y - alien['size'][1]/2, *alien['size'])
            if player_rect.colliderect(alien_rect) or alien['pos'].y > self.HEIGHT - 20:
                self.game_over = True
                reward -= 10 # Heavy penalty for letting aliens pass
                break
        
        return reward

    def _create_explosion(self, pos, colors):
        for _ in range(30):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)),
                'lifespan': self.np_random.integers(10, 25),
                'radius': self.np_random.uniform(1, 4),
                'color': random.choice(colors)
            })

    def _check_progression(self):
        reward = 0
        if not self.aliens and not self.game_over:
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self._setup_stage()
                reward += 50
            else:
                self.game_won = True
                self.game_over = True
                reward += 100
        return reward

    def _check_termination(self):
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for star in self.stars:
            star['pos'].y += star['speed']
            if star['pos'].y > self.HEIGHT:
                star['pos'].y = 0
                star['pos'].x = self.np_random.uniform(0, self.WIDTH)
            
            color_val = int(star['speed'] * 100)
            color = (color_val, color_val, color_val + 50)
            pygame.draw.circle(self.screen, color, (int(star['pos'].x), int(star['pos'].y)), star['radius'])

    def _render_game(self):
        if not self.game_over or self.game_won:
            self._render_player()
        
        self._render_aliens()
        self._render_projectiles()
        self._render_particles()

    def _render_player(self):
        if self.player_lives <= 0: return

        x, y = int(self.player_pos.x), int(self.player_pos.y)
        w, h = self.PLAYER_SIZE
        
        # Ship body
        points = [(x, y - h/2), (x - w/2, y + h/2), (x + w/2, y + h/2)]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Engine glow
        engine_y = y + h/2
        for i in range(5):
            alpha = 150 - i * 30
            radius = w/4 - i
            color = (*self.COLOR_PLAYER_GLOW, alpha)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (x - radius, engine_y + i*2))

    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'].x), int(alien['pos'].y)
            w, h = alien['size']
            color = self._get_alien_info(alien['type'])['color']
            
            rect = (x - w/2, y - h/2, w, h)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            # Pulsating glow
            glow_alpha = 100 + math.sin(pygame.time.get_ticks() / 200) * 50
            glow_color = (*color, glow_alpha)
            glow_size = int(w * 1.4)
            temp_surf = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, glow_color, (0, 0, glow_size, glow_size), border_radius=5)
            self.screen.blit(temp_surf, (x - glow_size/2, y - glow_size/2), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_projectiles(self):
        for proj in self.projectiles:
            x, y = int(proj['pos'].x), int(proj['pos'].y)
            w, h = proj['size']
            rect = (x - w/2, y - h/2, w, h)
            pygame.draw.rect(self.screen, proj['color'], rect, border_radius=2)
            
            # Trail effect
            glow_color = (*proj['color'], 100)
            glow_size_w, glow_size_h = int(w * 2), int(h * 1.5)
            temp_surf = pygame.Surface((glow_size_w, glow_size_h), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, glow_color, (0,0, glow_size_w, glow_size_h), border_radius=3)
            self.screen.blit(temp_surf, (x - glow_size_w/2, y - glow_size_h/2), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / 25.0))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (p['pos'].x - p['radius'], p['pos'].y - p['radius']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage
        stage_romans = {1: 'I', 2: 'II', 3: 'III'}
        stage_text = self.font_small.render(f"STAGE: {stage_romans.get(self.stage, 'I')}", True, self.COLOR_TEXT)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, 10))

        # Timer
        timer_secs = self.stage_timer // self.FPS
        timer_text = self.font_small.render(f"TIME: {timer_secs}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.WIDTH/2 - timer_text.get_width()/2, 35))

        # Lives
        for i in range(self.player_lives):
            x, y = self.WIDTH - 20 - (i * 25), 20
            w, h = self.PLAYER_SIZE
            points = [(x, y - h/2), (x - w/2 + 5, y + h/2), (x + w/2 - 5, y + h/2)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Game Over / Win Message
        if self.game_over:
            if self.game_won:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.WIDTH/2 - end_text.get_width()/2, self.HEIGHT/2 - end_text.get_height()/2))
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "stage": self.stage,
        }
    
    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not run in a headless environment
    try:
        del os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Shooter")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Keyboard controls
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            terminated = False
            total_reward = 0

    env.close()