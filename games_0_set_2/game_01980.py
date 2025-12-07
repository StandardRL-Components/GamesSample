
# Generated: 2025-08-27T18:52:33.254987
# Source Brief: brief_01980.md
# Brief Index: 1980

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Collect yellow gems and avoid the enemies."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect glittering gems while dodging cunning enemies in a vibrant, top-down arcade environment."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 75, 128)
    COLOR_GEM = (255, 220, 0)
    COLOR_GEM_GLOW = (128, 110, 0)
    COLOR_ENEMY_1 = (255, 50, 50)
    COLOR_ENEMY_2 = (255, 150, 50)
    COLOR_ENEMY_3 = (200, 50, 255)
    COLOR_TEXT = (220, 220, 220)
    
    # Game Parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 1000
    WIN_SCORE = 50
    STARTING_LIVES = 3
    PLAYER_SPEED = 6
    PLAYER_RADIUS = 12
    GEM_RADIUS = 10
    ENEMY_RADIUS = 14
    NUM_ENEMIES = 3
    INITIAL_ENEMY_SPEED = 1.5
    RISKY_PLAY_RADIUS = 100 # How close to an enemy for a "risky" gem grab

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        self.render_mode = render_mode
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Game state variables (initialized in reset)
        self.player_pos = None
        self.player_lives = None
        self.gems_collected = None
        self.gem_pos = None
        self.enemies = []
        self.particles = []
        self.current_enemy_speed = None
        self.invincibility_timer = 0
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.gems_collected = 0
        self.game_over = False
        self.player_lives = self.STARTING_LIVES
        self.invincibility_timer = 0
        self.current_enemy_speed = self.INITIAL_ENEMY_SPEED
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self._spawn_gem()
        self._spawn_enemies()
        
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        
        # --- Pre-action state ---
        dist_to_gem_before = self._get_distance(self.player_pos, self.gem_pos)

        # --- Update Game Logic ---
        self._handle_player_movement(movement)
        self._update_enemies()
        self._update_particles()
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1

        # --- Calculate Rewards & Handle Interactions ---
        reward = 0
        
        # 1. Movement reward
        dist_to_gem_after = self._get_distance(self.player_pos, self.gem_pos)
        if dist_to_gem_after < dist_to_gem_before:
            reward += 1.0  # Closer to gem
        else:
            reward -= 0.1 # Further from gem
        
        # 2. Gem collection
        if dist_to_gem_after < self.PLAYER_RADIUS + self.GEM_RADIUS:
            # Sfx: GEM_COLLECT
            self.gems_collected += 1
            self.score += 10
            dist_to_nearest_enemy = min(self._get_distance(self.player_pos, e['pos']) for e in self.enemies)
            
            if dist_to_nearest_enemy < self.RISKY_PLAY_RADIUS:
                reward += 15.0 # Base + risky bonus = 10 + 5
            else:
                reward -= 10.0 # Base - safe penalty = 10 - 20
            
            self._create_particles(self.gem_pos, self.COLOR_GEM, 30)
            self._spawn_gem()
            
            # Difficulty scaling
            if self.gems_collected > 0 and self.gems_collected % 10 == 0:
                self.current_enemy_speed += 0.05

        # 3. Enemy collision
        if self.invincibility_timer == 0:
            for enemy in self.enemies:
                if self._get_distance(self.player_pos, enemy['pos']) < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                    # Sfx: PLAYER_HIT
                    self.player_lives -= 1
                    reward -= 30.0
                    self.invincibility_timer = 60 # 2 seconds of invincibility
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 20)
                    self._create_particles(enemy['pos'], enemy['color'], 20)
                    break # Only one hit per frame

        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if self.player_lives <= 0:
            reward -= 100.0
            terminated = True
            self.game_over = True
        elif self.gems_collected >= self.WIN_SCORE:
            reward += 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_particles()
        self._render_enemies()
        self._render_gem()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "gems_collected": self.gems_collected,
        }
        
    # --- Helper Functions ---

    def _spawn_gem(self):
        # Ensure gem doesn't spawn too close to the player or enemies
        while True:
            pos = self.np_random.uniform(
                low=[self.GEM_RADIUS, self.GEM_RADIUS],
                high=[self.SCREEN_WIDTH - self.GEM_RADIUS, self.SCREEN_HEIGHT - self.GEM_RADIUS]
            ).astype(np.float32)
            
            if self._get_distance(pos, self.player_pos) < 100:
                continue
            
            too_close_to_enemy = False
            for enemy in self.enemies:
                if self._get_distance(pos, enemy['pos']) < 50:
                    too_close_to_enemy = True
                    break
            if not too_close_to_enemy:
                self.gem_pos = pos
                break

    def _spawn_enemies(self):
        self.enemies = []
        colors = [self.COLOR_ENEMY_1, self.COLOR_ENEMY_2, self.COLOR_ENEMY_3]
        patterns = ['circular', 'horizontal', 'vertical']
        self.np_random.shuffle(patterns)

        for i in range(self.NUM_ENEMIES):
            # Spawn enemies in different quadrants
            quadrant_x = self.SCREEN_WIDTH / 2 * (i % 2)
            quadrant_y = self.SCREEN_HEIGHT / 2 * (i // 2)
            
            pos = np.array([
                quadrant_x + self.np_random.uniform(50, self.SCREEN_WIDTH/2 - 50),
                quadrant_y + self.np_random.uniform(50, self.SCREEN_HEIGHT/2 - 50)
            ], dtype=np.float32)
            
            enemy = {
                'pos': pos,
                'color': colors[i % len(colors)],
                'pattern': patterns[i % len(patterns)],
                'angle': self.np_random.uniform(0, 2 * math.pi),
                'direction': self.np_random.choice([-1, 1]),
                'center': pos.copy(),
                'patrol_range': self.np_random.uniform(50, 100)
            }
            self.enemies.append(enemy)

    def _handle_player_movement(self, movement):
        velocity = np.zeros(2, dtype=np.float32)
        if movement == 1: # Up
            velocity[1] = -self.PLAYER_SPEED
        elif movement == 2: # Down
            velocity[1] = self.PLAYER_SPEED
        elif movement == 3: # Left
            velocity[0] = -self.PLAYER_SPEED
        elif movement == 4: # Right
            velocity[0] = self.PLAYER_SPEED
        
        self.player_pos += velocity
        
        # Clamp player position to screen bounds
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy['pattern'] == 'circular':
                enemy['angle'] += 0.02 * self.current_enemy_speed
                radius = enemy['patrol_range']
                enemy['pos'][0] = enemy['center'][0] + radius * math.cos(enemy['angle'])
                enemy['pos'][1] = enemy['center'][1] + radius * math.sin(enemy['angle'])
            elif enemy['pattern'] == 'horizontal':
                enemy['pos'][0] += self.current_enemy_speed * enemy['direction']
                if abs(enemy['pos'][0] - enemy['center'][0]) > enemy['patrol_range']:
                    enemy['direction'] *= -1
            elif enemy['pattern'] == 'vertical':
                enemy['pos'][1] += self.current_enemy_speed * enemy['direction']
                if abs(enemy['pos'][1] - enemy['center'][1]) > enemy['patrol_range']:
                    enemy['direction'] *= -1
            
            # Keep enemies on screen
            enemy['pos'][0] = np.clip(enemy['pos'][0], self.ENEMY_RADIUS, self.SCREEN_WIDTH - self.ENEMY_RADIUS)
            enemy['pos'][1] = np.clip(enemy['pos'][1], self.ENEMY_RADIUS, self.SCREEN_HEIGHT - self.ENEMY_RADIUS)


    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=np.float32)
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': self.np_random.integers(20, 40),
                'color': color,
                'radius': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        
    def _get_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    # --- Rendering Functions ---

    def _render_player(self):
        pos_int = self.player_pos.astype(int)
        
        # Flash if invincible
        if self.invincibility_timer > 0 and self.invincibility_timer % 10 < 5:
            return

        # Draw glow
        glow_radius = int(self.PLAYER_RADIUS * 1.8)
        for i in range(glow_radius, self.PLAYER_RADIUS, -1):
            alpha = int(30 * (1 - (i - self.PLAYER_RADIUS) / (glow_radius - self.PLAYER_RADIUS)))
            color = (*self.COLOR_PLAYER_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], i, color)
        
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_gem(self):
        pos_int = self.gem_pos.astype(int)
        
        # Pulsating effect
        pulse = (math.sin(self.steps * 0.1) + 1) / 2
        current_radius = int(self.GEM_RADIUS * (0.9 + pulse * 0.2))
        
        # Draw glow
        glow_radius = int(current_radius * 2.0)
        for i in range(glow_radius, current_radius, -1):
            alpha = int(40 * (1 - (i - current_radius) / (glow_radius - current_radius)))
            color = (*self.COLOR_GEM_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], i, color)
        
        pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], current_radius, self.COLOR_GEM)
        pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], current_radius, self.COLOR_GEM)

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_int = enemy['pos'].astype(int)
            rotation_angle = self.steps * 0.05
            
            # Draw a rotating triangle for enemies
            points = []
            for i in range(3):
                angle = rotation_angle + (i * 2 * math.pi / 3)
                x = pos_int[0] + self.ENEMY_RADIUS * math.cos(angle)
                y = pos_int[1] + self.ENEMY_RADIUS * math.sin(angle)
                points.append((int(x), int(y)))
                
            pygame.gfxdraw.aapolygon(self.screen, points, enemy['color'])
            pygame.gfxdraw.filled_polygon(self.screen, points, enemy['color'])

    def _render_particles(self):
        for p in self.particles:
            pos_int = p['pos'].astype(int)
            alpha = int(255 * (p['lifespan'] / 40.0))
            color_with_alpha = (*p['color'], alpha)
            radius = int(p['radius'] * (p['lifespan'] / 40.0))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], max(0, radius), color_with_alpha)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Gems collected
        gem_text = self.font_small.render(f"GEMS: {self.gems_collected}/{self.WIN_SCORE}", True, self.COLOR_TEXT)
        self.screen.blit(gem_text, (10, 40))

        # Lives
        for i in range(self.player_lives):
            pos = (self.SCREEN_WIDTH - 30 - i * (self.PLAYER_RADIUS * 2 + 5), 25)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.gems_collected >= self.WIN_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

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
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'x11' or 'windows' or 'mac' depending on your OS
    
    env = GameEnv(render_mode="human")
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Create a window for human rendering
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Collector")
    clock = pygame.time.Clock()
    
    terminated = False
    while not terminated:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the human-display screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()