import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move. Hold space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Pilot a spaceship in a top-down arcade shooter to defeat waves of procedurally generated alien invaders."
    )

    # Should frames auto-advance or wait for user input?
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
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 128, 64)
        self.COLOR_BULLET_PLAYER = (255, 255, 255)
        self.COLOR_BULLET_ENEMY = (255, 0, 255)
        self.COLOR_ALIEN_RED = (255, 50, 50)
        self.COLOR_ALIEN_BLUE = (50, 150, 255)
        self.COLOR_ALIEN_YELLOW = (255, 255, 50)
        self.COLOR_PARTICLE = (255, 165, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEART = (255, 0, 0)
        
        # Game constants
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 12
        self.BULLET_SPEED = 8
        self.MAX_PLAYER_BULLETS = 5
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.MAX_STAGES = 3
        self.MAX_STEPS = self.MAX_STAGES * 60 * 30 # 3 stages * 60s * 30fps
        self.STAGE_DURATION_FRAMES = 60 * 30 # 60s * 30fps

        # Initialize state variables
        self.aliens = []
        self.reset()
        
        # self.validate_implementation() # This can be commented out for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.stage = 1
        self.stage_timer = self.STAGE_DURATION_FRAMES

        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 40]
        self.player_health = 3
        
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []
        self._spawn_aliens()

        self.last_space_held = False
        self.player_fire_timer = 0
        
        self.stars = []
        for _ in range(150):
            self.stars.append({
                "x": self.np_random.uniform(0, self.WIDTH),
                "y": self.np_random.uniform(0, self.HEIGHT),
                "z": self.np_random.uniform(0.1, 1.0)
            })

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.clock.tick(30)
        reward = 0.0
        
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held)

        self._update_bullets()
        self._update_aliens()
        self._update_particles()
        self._update_stars()

        reward += self._handle_collisions()

        stage_clear_reward = self._check_stage_clear()
        reward += stage_clear_reward
        
        self.steps += 1
        self.stage_timer -= 1
        
        terminated = False
        if self.player_health <= 0:
            terminated = True
        if self.stage_timer <= 0 and self.stage < self.MAX_STAGES:
            terminated = True
            reward -= 10
        if self.steps >= self.MAX_STEPS:
            terminated = True
        if stage_clear_reward == 100: # Game won
            terminated = True
            
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_held):
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: self.player_pos[0] += self.PLAYER_SPEED
            
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
            
        if space_held and self.player_fire_timer == 0 and len(self.player_bullets) < self.MAX_PLAYER_BULLETS:
            self.player_bullets.append({"pos": [self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE], "speed": -self.BULLET_SPEED})
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_bullets(self):
        for bullet in self.player_bullets[:]:
            bullet["pos"][1] += bullet["speed"]
            if bullet["pos"][1] < 0: self.player_bullets.remove(bullet)

        for bullet in self.enemy_bullets[:]:
            bullet["pos"][1] += bullet["speed"]
            if bullet["pos"][1] > self.HEIGHT: self.enemy_bullets.remove(bullet)

    def _update_aliens(self):
        alien_speed_mod = 1.0 + (self.stage - 1) * 0.05
        
        for alien in self.aliens:
            if alien["type"] == 'red':
                alien['pos'][0] += 2 * alien_speed_mod * alien['move_dir']
                if not (20 < alien['pos'][0] < self.WIDTH - 20):
                    alien['move_dir'] *= -1
                    alien['pos'][1] += 15
            elif alien['type'] == 'blue':
                alien['move_angle'] += 0.05 * alien_speed_mod
                alien['pos'][0] += math.cos(alien['move_angle']) * 1.5 * alien_speed_mod
                alien['pos'][1] += math.sin(alien['move_angle']) * 0.5 * alien_speed_mod
            elif alien['type'] == 'yellow':
                alien['pos'][1] += 3 * alien_speed_mod
                if alien['pos'][1] > self.HEIGHT:
                    alien['pos'][0] = self.np_random.uniform(40, self.WIDTH - 40)
                    alien['pos'][1] = -20
            
            fire_rate_mod = max(1, 1 * (self.stage - 1))
            
            alien['fire_timer'] += 1
            fire_cooldowns = {'red': 30, 'blue': 20, 'yellow': 10}
            if alien['fire_timer'] > (fire_cooldowns[alien['type']] - fire_rate_mod):
                alien['fire_timer'] = 0
                self.enemy_bullets.append({"pos": list(alien["pos"]), "speed": self.BULLET_SPEED * 0.6})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

    def _update_stars(self):
        for star in self.stars:
            star['y'] += star['z'] * 0.5
            if star['y'] > self.HEIGHT:
                star['y'] = 0
                star['x'] = self.np_random.uniform(0, self.WIDTH)

    def _handle_collisions(self):
        reward = 0.0
        for bullet in self.player_bullets[:]:
            for alien in self.aliens[:]:
                if math.hypot(bullet['pos'][0] - alien['pos'][0], bullet['pos'][1] - alien['pos'][1]) < alien['size'] + 4:
                    self._create_explosion(alien['pos'])
                    self.aliens.remove(alien)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    reward += 1.0
                    self.score += 100
                    break
        
        for bullet in self.enemy_bullets[:]:
            if math.hypot(bullet['pos'][0] - self.player_pos[0], bullet['pos'][1] - self.player_pos[1]) < self.PLAYER_SIZE:
                self.enemy_bullets.remove(bullet)
                self._create_explosion(self.player_pos, small=True)
                self.player_health -= 1
                reward -= 1.0 # Lost a life
                reward -= 0.01 # Got hit
                if self.player_health <= 0: self.game_over = True
                break
        return reward

    def _create_explosion(self, pos, small=False):
        num_particles = 10 if small else 30
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'radius': self.np_random.uniform(1, 3)
            })

    def _check_stage_clear(self):
        if not self.aliens:
            if self.stage < self.MAX_STAGES:
                self.stage += 1
                self._spawn_aliens()
                self.stage_timer = self.STAGE_DURATION_FRAMES
                self.player_bullets.clear()
                self.enemy_bullets.clear()
                return 10.0
            else:
                return 100.0
        return 0.0

    def _spawn_aliens(self):
        self.aliens.clear()
        num_aliens = 8 + self.stage * 2
        
        for i in range(num_aliens):
            if self.stage == 1: alien_type = 'red'
            elif self.stage == 2: alien_type = self.np_random.choice(['red', 'blue'], p=[0.7, 0.3])
            else: alien_type = self.np_random.choice(['red', 'blue', 'yellow'], p=[0.4, 0.4, 0.2])
            
            spawn_x = self.np_random.uniform(40, self.WIDTH - 40)
            spawn_y = self.np_random.uniform(40, self.HEIGHT / 2)
            
            self.aliens.append({
                "pos": [spawn_x, spawn_y], "type": alien_type, "size": 10,
                "fire_timer": self.np_random.integers(0, 30),
                "move_dir": 1 if self.np_random.random() > 0.5 else -1,
                "move_angle": self.np_random.uniform(0, 2 * math.pi)
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "stage": self.stage, "health": self.player_health}

    def _render_stars(self):
        for star in self.stars:
            size = int(star['z'] * 2)
            color_val = int(star['z'] * 100) + 50
            pygame.draw.rect(self.screen, (color_val, color_val, color_val), (int(star['x']), int(star['y']), size, size))

    def _render_game(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*self.COLOR_PARTICLE, alpha))
            except TypeError: # Sometimes alpha is invalid
                pass


        for alien in self.aliens:
            pos_int = (int(alien['pos'][0]), int(alien['pos'][1]))
            s = alien['size']
            if alien['type'] == 'red':
                pygame.draw.rect(self.screen, self.COLOR_ALIEN_RED, (pos_int[0] - s, pos_int[1] - s, s*2, s*2))
            elif alien['type'] == 'blue':
                pts = [(pos_int[0] + s * math.cos(i*math.pi/3), pos_int[1] + s * math.sin(i*math.pi/3)) for i in range(6)]
                pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_ALIEN_BLUE)
            elif alien['type'] == 'yellow':
                pts = [(pos_int[0], pos_int[1]-s), (pos_int[0]+s, pos_int[1]), (pos_int[0], pos_int[1]+s), (pos_int[0]-s, pos_int[1])]
                pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_ALIEN_YELLOW)
        
        for bullet in self.player_bullets:
            pos_int = (int(bullet['pos'][0]), int(bullet['pos'][1]))
            pygame.draw.rect(self.screen, self.COLOR_BULLET_PLAYER, (pos_int[0]-1, pos_int[1]-4, 3, 8))

        for bullet in self.enemy_bullets:
            pos_int = (int(bullet['pos'][0]), int(bullet['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], 4, self.COLOR_BULLET_ENEMY)

        if self.player_health > 0:
            pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
            s = self.PLAYER_SIZE
            pts = [(pos_int[0], pos_int[1]-s), (pos_int[0]-s/2, pos_int[1]+s/2), (pos_int[0]+s/2, pos_int[1]+s/2)]
            glow_pts = [(pos_int[0], pos_int[1]-s*1.5), (pos_int[0]-s, pos_int[1]+s), (pos_int[0]+s, pos_int[1]+s)]
            try:
                pygame.gfxdraw.filled_polygon(self.screen, glow_pts, (*self.COLOR_PLAYER_GLOW, 100))
            except TypeError: # Sometimes color is invalid
                pass
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_PLAYER)
            pygame.gfxdraw.aapolygon(self.screen, pts, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        stage_text = self.font_ui.render(f"STAGE: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.WIDTH/2 - stage_text.get_width()/2, 10))
        
        for i in range(self.player_health):
            pos = (20 + i * 25, 20)
            pts = [(pos[0], pos[1]-5), (pos[0]+5, pos[1]-10), (pos[0]+10, pos[1]-5), (pos[0]+10, pos[1]), (pos[0], pos[1]+10), (pos[0]-10, pos[1]), (pos[0]-10, pos[1]-5), (pos[0]-5, pos[1]-10)]
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_HEART)

        time_left_sec = max(0, self.stage_timer // 30)
        timer_text = self.font_ui.render(f"TIME: {time_left_sec}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (10, 40))

        if self.game_over:
            msg = "YOU WIN!" if self.player_health > 0 and self.stage > self.MAX_STAGES else "GAME OVER"
            over_text = self.font_game_over.render(msg, True, (255, 255, 255))
            self.screen.blit(over_text, over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2)))

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")