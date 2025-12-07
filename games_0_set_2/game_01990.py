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


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move your character (blue square)."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "Collect 20 fruits (colored circles) while dodging the red enemy triangles. Get bonus points for risky collections near enemies!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_FRUITS = 20
        
        # Player
        self.PLAYER_SPEED = 5
        self.PLAYER_SIZE = 12

        # Enemy
        self.ENEMY_COUNT = 3
        self.ENEMY_SIZE = 14
        self.ENEMY_SPEED = 2.5
        self.DANGER_PROXIMITY_BONUS = 80 # pixels

        # Fruit
        self.FRUIT_COUNT = 5
        self.FRUIT_SIZE = 8

        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_FRUITS = [
            (255, 230, 0), (255, 128, 0), (50, 200, 50), 
            (200, 0, 200), (0, 200, 200)
        ]
        self.COLOR_TEXT = (240, 240, 240)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        # Headless setup
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)

        # --- Internal State ---
        # These are initialized in reset()
        self.np_random = None
        self.player_pos = None
        self.enemies = []
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        
        # self.reset() is called by the environment wrapper, no need to call it here.
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        
        self.enemies = self._spawn_enemies()
        self.fruits = []
        for _ in range(self.FRUIT_COUNT):
            self._spawn_fruit()
            
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        # --- Pre-move state for reward calculation ---
        prev_dist_to_fruit = self._get_closest_distance(self.fruits)
        prev_dist_to_enemy = self._get_closest_distance(self.enemies)

        # --- Update Game Logic ---
        self._update_player(movement)
        self._update_enemies()
        self._update_particles()
        
        # --- Post-move state for reward calculation ---
        curr_dist_to_fruit = self._get_closest_distance(self.fruits)
        curr_dist_to_enemy = self._get_closest_distance(self.enemies)

        # --- Calculate Reward ---
        reward = 0
        
        # Continuous reward for moving towards fruit
        if curr_dist_to_fruit < prev_dist_to_fruit:
            reward += 1.0
        # Penalty for moving away from a distant fruit
        elif curr_dist_to_fruit > prev_dist_to_fruit and prev_dist_to_fruit > 150:
            reward -= 2.0

        # Continuous penalty for moving towards enemy
        if curr_dist_to_enemy < prev_dist_to_enemy:
            reward -= 0.1

        # Event-based rewards from collisions
        reward += self._handle_collisions()

        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        # Terminal rewards override others
        if terminated and not truncated:
            if self.fruits_collected >= self.WIN_CONDITION_FRUITS:
                reward = 100.0
                self.score += 100 # Add final bonus to score
            elif self.game_over: # Hit an enemy
                reward = -100.0

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, movement):
        if movement == 1:  # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.SCREEN_WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.SCREEN_HEIGHT - self.PLAYER_SIZE)

    def _update_enemies(self):
        for enemy in self.enemies:
            enemy['timer'] += 0.05
            if enemy['pattern'] == 'circle':
                enemy['pos'][0] = enemy['center'][0] + math.cos(enemy['timer']) * enemy['radius']
                enemy['pos'][1] = enemy['center'][1] + math.sin(enemy['timer']) * enemy['radius']
            elif enemy['pattern'] == 'figure-eight':
                enemy['pos'][0] = enemy['center'][0] + math.sin(enemy['timer']) * enemy['radius']
                enemy['pos'][1] = enemy['center'][1] + math.sin(enemy['timer'] * 2) * enemy['radius'] * 0.6
            elif enemy['pattern'] == 'bounce':
                enemy['pos'] += enemy['vel'] * self.ENEMY_SPEED
                if enemy['pos'][0] <= self.ENEMY_SIZE or enemy['pos'][0] >= self.SCREEN_WIDTH - self.ENEMY_SIZE:
                    enemy['vel'][0] *= -1
                if enemy['pos'][1] <= self.ENEMY_SIZE or enemy['pos'][1] >= self.SCREEN_HEIGHT - self.ENEMY_SIZE:
                    enemy['vel'][1] *= -1

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['radius'] = max(0, p['radius'] - 0.2)

    def _handle_collisions(self):
        reward = 0
        # Player-Fruit
        for fruit in self.fruits[:]:
            dist = np.linalg.norm(self.player_pos - fruit['pos'])
            if dist < self.PLAYER_SIZE + self.FRUIT_SIZE:
                self.fruits.remove(fruit)
                self._spawn_fruit()
                self.fruits_collected += 1
                
                event_reward = 10.0
                self.score += 10

                # Check for proximity bonus
                dist_to_enemy = self._get_closest_distance(self.enemies)
                if dist_to_enemy < self.DANGER_PROXIMITY_BONUS:
                    event_reward += 5.0
                    self.score += 5
                
                reward += event_reward
                self._create_particles(self.player_pos, fruit['color'])
                break # Only collect one fruit per step

        # Player-Enemy
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + self.ENEMY_SIZE - 4: # -4 for tighter hitbox
                self.game_over = True
                self.score -= 100
                break
        
        return reward

    def _check_termination(self):
        return self.game_over or self.fruits_collected >= self.WIN_CONDITION_FRUITS

    def _get_closest_distance(self, entities):
        if not entities:
            return float('inf')
        
        min_dist = float('inf')
        for entity in entities:
            dist = np.linalg.norm(self.player_pos - entity['pos'])
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def _spawn_enemies(self):
        enemies = []
        w, h = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        
        # Enemy 1: Circular path
        enemies.append({
            'pos': np.array([0,0], dtype=np.float32), 'pattern': 'circle', 
            'center': np.array([w*0.25, h*0.5]), 'radius': w*0.15, 'timer': self.np_random.random() * 10
        })
        # Enemy 2: Figure-eight path
        enemies.append({
            'pos': np.array([0,0], dtype=np.float32), 'pattern': 'figure-eight', 
            'center': np.array([w*0.75, h*0.5]), 'radius': w*0.18, 'timer': self.np_random.random() * 10
        })
        # Enemy 3: Bouncing path
        angle = self.np_random.uniform(0, 2 * math.pi)
        enemies.append({
            'pos': np.array([w*0.5, h*0.2], dtype=np.float32), 'pattern': 'bounce', 
            'vel': np.array([math.cos(angle), math.sin(angle)], dtype=np.float32),
            'timer': self.np_random.random() * 10 # FIX: Added missing timer key
        })
        return enemies

    def _spawn_fruit(self):
        while True:
            pos = self.np_random.uniform(
                [self.FRUIT_SIZE, self.FRUIT_SIZE],
                [self.SCREEN_WIDTH - self.FRUIT_SIZE, self.SCREEN_HEIGHT - self.FRUIT_SIZE]
            ).astype(np.float32)
            
            # Ensure it doesn't spawn too close to the player
            if self.player_pos is not None and np.linalg.norm(pos - self.player_pos) < 50:
                continue
            
            # Ensure it doesn't spawn on other fruits
            too_close = False
            for fruit in self.fruits:
                if np.linalg.norm(pos - fruit['pos']) < self.FRUIT_SIZE * 3:
                    too_close = True
                    break
            if not too_close:
                break
        
        color_idx = self.np_random.integers(0, len(self.COLOR_FRUITS))
        color = self.COLOR_FRUITS[color_idx]
        self.fruits.append({'pos': pos, 'color': color})

    def _create_particles(self, pos, color):
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'radius': self.np_random.uniform(2, 5),
                'lifetime': self.np_random.integers(10, 20),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'].astype(int), int(p['radius']))

        # Fruits
        for fruit in self.fruits:
            pos = fruit['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FRUIT_SIZE, fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FRUIT_SIZE, fruit['color'])

        # Enemies
        for enemy in self.enemies:
            self._draw_rotated_triangle(self.screen, self.COLOR_ENEMY, enemy['pos'], self.ENEMY_SIZE, enemy['timer'])

        # Player
        if self.player_pos is not None:
            pos = self.player_pos.astype(int)
            glow_radius = int(self.PLAYER_SIZE * 1.8)
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            player_rect = pygame.Rect(pos[0] - self.PLAYER_SIZE, pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _draw_rotated_triangle(self, surface, color, center, size, angle):
        points = []
        for i in range(3):
            point_angle = angle + (i * 2 * math.pi / 3)
            x = center[0] + size * 1.5 * math.cos(point_angle)
            y = center[1] + size * 1.5 * math.sin(point_angle)
            points.append((int(x), int(y)))
        
        if len(points) == 3:
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

    def _render_ui(self):
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        fruits_text = self.font_ui.render(f"FRUITS: {self.fruits_collected}/{self.WIN_CONDITION_FRUITS}", True, self.COLOR_TEXT)
        self.screen.blit(fruits_text, (10, 35))

        if self._check_termination() or self.steps >= self.MAX_STEPS:
            msg = ""
            color = self.COLOR_TEXT
            if self.fruits_collected >= self.WIN_CONDITION_FRUITS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            elif self.game_over:
                msg = "GAME OVER"
                color = (255, 100, 100)
            elif self.steps >= self.MAX_STEPS:
                msg = "TIME UP"
                color = (255, 255, 100)
            
            if msg:
                end_text = self.font_game_over.render(msg, True, color)
                text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
                self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_collected": self.fruits_collected,
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # Note: Requires a display. Will not run in a headless environment.
    import os
    # If on a headless server, you might need:
    os.environ.pop("SDL_VIDEODRIVER", None)

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Arcade Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(seed=random.randint(0, 10000))
                action.fill(0)
        
        # --- Human Controls ---
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
        
        action = np.array([movement, space_held, shift_held])
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}, Term: {terminated}, Trunc: {truncated}")
            # The game will freeze on the 'Game Over' screen. Press 'R' to reset.

        # --- Render to Display ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Frame Rate ---
        clock.tick(30) # Run at 30 FPS

    env.close()