import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set the SDL_VIDEODRIVER to "dummy" for headless operation
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Collect 50 fruits to win. Avoid the enemies!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect 50 pieces of fruit while dodging increasingly aggressive enemies in this fast-paced, top-down arcade game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1500 # 50 seconds at 30fps
        self.FRUITS_TO_WIN = 50
        self.NUM_FRUITS_ON_SCREEN = 5
        self.NUM_ENEMIES_PATROL = 2
        self.NUM_ENEMIES_CHASE = 1
        self.NUM_ENEMIES_RANDOM = 2
        
        # Player constants
        self.PLAYER_SIZE = 12
        self.PLAYER_ACCEL = 0.8
        self.PLAYER_FRICTION = 0.92
        
        # Enemy constants
        self.BASE_ENEMY_SPEED = 1.0
        self.ENEMY_SPEED_INCREASE = 0.05
        self.ENEMY_SIZE = 10
        self.DANGER_RADIUS = 80 # For bonus points

        # Fruit constants
        self.FRUIT_SIZE = 6

        # Colors
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_BOUNDARY = (50, 50, 70)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (0, 255, 128, 50)
        self.COLOR_ENEMY_PATROL = (255, 50, 50)
        self.COLOR_ENEMY_CHASE = (50, 150, 255)
        self.COLOR_ENEMY_RANDOM = (255, 255, 0)
        self.FRUIT_COLORS = [(255, 0, 255), (0, 255, 255), (255, 128, 0)]
        self.COLOR_UI_TEXT = (220, 220, 240)
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_end = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Initialize state variables
        self.player_pos = None
        self.player_vel = None
        self.enemies = []
        self.fruits = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        self.win = False
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.fruits_collected = 0
        self.game_over = False
        self.win = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)

        self.enemies = []
        self.fruits = []
        self.particles = []
        
        # Spawn initial enemies
        for _ in range(self.NUM_ENEMIES_PATROL):
            self._spawn_enemy('patrol')
        for _ in range(self.NUM_ENEMIES_CHASE):
            self._spawn_enemy('chase')
        for _ in range(self.NUM_ENEMIES_RANDOM):
            self._spawn_enemy('random')

        # Spawn initial fruits
        for _ in range(self.NUM_FRUITS_ON_SCREEN):
            self._spawn_fruit()
        
        return self._get_observation(), self._get_info()

    def _spawn_enemy(self, type):
        padding = 50
        # Ensure enemies spawn at a safe distance from the player's starting point
        while True:
            pos = np.array([
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            ], dtype=np.float32)
            if np.linalg.norm(pos - self.player_pos) > 150:
                break
        
        enemy = {'pos': pos, 'type': type, 'size': self.ENEMY_SIZE}
        if type == 'patrol':
            enemy['path'] = [
                np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)])
                for _ in range(self.np_random.integers(2, 5))
            ]
            enemy['path_index'] = 0
        elif type == 'random':
            enemy['change_dir_timer'] = 0
            enemy['vel'] = np.zeros(2)
        
        self.enemies.append(enemy)

    def _spawn_fruit(self):
        padding = 20
        while True:
            pos = np.array([
                self.np_random.uniform(padding, self.WIDTH - padding),
                self.np_random.uniform(padding, self.HEIGHT - padding)
            ], dtype=np.float32)
            
            # Ensure it doesn't spawn too close to the player
            if np.linalg.norm(pos - self.player_pos) < 50:
                continue

            # Ensure it doesn't spawn too close to other fruits
            too_close = False
            for fruit in self.fruits:
                if np.linalg.norm(pos - fruit['pos']) < 20:
                    too_close = True
                    break
            if not too_close:
                break
        
        color_index = self.np_random.integers(len(self.FRUIT_COLORS))
        color = self.FRUIT_COLORS[color_index]
        self.fruits.append({'pos': pos, 'color': color, 'size': self.FRUIT_SIZE})

    def step(self, action):
        reward = -0.01  # Small penalty for every step to encourage efficiency
        
        if not self.game_over:
            # --- Player Logic ---
            dist_before = self._get_dist_to_nearest_fruit()
            self._update_player(action)
            dist_after = self._get_dist_to_nearest_fruit()

            # Reward for moving towards fruit
            if dist_after < dist_before:
                reward += 0.1
            else:
                reward -= 0.02
            
            # --- Enemy Logic ---
            self._update_enemies()

            # --- Collision & Interaction Logic ---
            reward += self._handle_collisions()

        # --- Update Particles ---
        self._update_particles()
        
        self.steps += 1
        
        # --- Termination Logic ---
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        
        if self.fruits_collected >= self.FRUITS_TO_WIN and not self.win:
            self.win = True
            self.game_over = True
            terminated = True
            reward = 100
            self.score += 100

        if self.game_over and not self.win and reward > -100: # Ensure we only apply death penalty once
            reward = -100
            self.score -= 100
        
        if terminated or truncated:
            # Game has ended, no further updates should happen.
            # We can just return the final state.
            pass

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_player(self, action):
        movement = action[0]
        acc = np.zeros(2, dtype=np.float32)
        if movement == 1: acc[1] -= self.PLAYER_ACCEL  # Up
        if movement == 2: acc[1] += self.PLAYER_ACCEL  # Down
        if movement == 3: acc[0] -= self.PLAYER_ACCEL  # Left
        if movement == 4: acc[0] += self.PLAYER_ACCEL  # Right
        
        self.player_vel += acc
        self.player_vel *= self.PLAYER_FRICTION
        self.player_pos += self.player_vel

        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_enemies(self):
        current_speed = self.BASE_ENEMY_SPEED + (self.fruits_collected // 10) * self.ENEMY_SPEED_INCREASE
        
        for enemy in self.enemies:
            if enemy['type'] == 'patrol':
                target = enemy['path'][enemy['path_index']]
                direction = target - enemy['pos']
                dist = np.linalg.norm(direction)
                if dist < 5:
                    enemy['path_index'] = (enemy['path_index'] + 1) % len(enemy['path'])
                else:
                    enemy['pos'] += (direction / dist) * current_speed
            
            elif enemy['type'] == 'chase':
                direction = self.player_pos - enemy['pos']
                dist = np.linalg.norm(direction)
                if dist > 0:
                    enemy['pos'] += (direction / dist) * (current_speed * 0.8) # Chasers are a bit slower
            
            elif enemy['type'] == 'random':
                enemy['change_dir_timer'] -= 1
                if enemy['change_dir_timer'] <= 0:
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    enemy['vel'] = np.array([math.cos(angle), math.sin(angle)]) * current_speed
                    enemy['change_dir_timer'] = self.np_random.integers(30, 90)
                enemy['pos'] += enemy['vel']

            # Enemy boundary checks
            enemy['pos'][0] = np.clip(enemy['pos'][0], enemy['size'], self.WIDTH - enemy['size'])
            enemy['pos'][1] = np.clip(enemy['pos'][1], enemy['size'], self.HEIGHT - enemy['size'])
            # Reverse velocity if hitting a wall for random enemies
            if enemy['type'] == 'random':
                if not (enemy['size'] < enemy['pos'][0] < self.WIDTH - enemy['size']):
                    enemy['vel'][0] *= -1
                if not (enemy['size'] < enemy['pos'][1] < self.HEIGHT - enemy['size']):
                    enemy['vel'][1] *= -1

    def _handle_collisions(self):
        reward = 0
        
        # Player-Fruit
        collected_indices = []
        for i, fruit in enumerate(self.fruits):
            dist = np.linalg.norm(self.player_pos - fruit['pos'])
            if dist < self.PLAYER_SIZE + fruit['size']:
                collected_indices.append(i)
                self.fruits_collected += 1
                
                reward += 10
                self.score += 10
                self._create_particles(fruit['pos'], fruit['color'], 15)

                # Proximity bonus
                if self._get_dist_to_nearest_enemy() < self.DANGER_RADIUS:
                    reward += 5
                    self.score += 5
        
        if collected_indices:
            self.fruits = [f for i, f in enumerate(self.fruits) if i not in collected_indices]
            for _ in range(len(collected_indices)):
                self._spawn_fruit()

        # Player-Enemy
        for enemy in self.enemies:
            dist = np.linalg.norm(self.player_pos - enemy['pos'])
            if dist < self.PLAYER_SIZE + enemy['size']:
                self.game_over = True
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 50, is_explosion=True)
                break
        
        return reward
        
    def _get_observation(self):
        # Auto-advance frame rate
        if self.auto_advance:
            self.clock.tick(30)

        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Boundaries
        pygame.draw.rect(self.screen, self.COLOR_BOUNDARY, (0, 0, self.WIDTH, self.HEIGHT), 4)

        # Particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], int(p['size']))

        # Fruits
        for fruit in self.fruits:
            pos = fruit['pos'].astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fruit['size'], fruit['color'])
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fruit['size'], fruit['color'])

        # Enemies
        for enemy in self.enemies:
            pos = enemy['pos'].astype(int)
            size = int(enemy['size'])
            if enemy['type'] == 'patrol':
                color = self.COLOR_ENEMY_PATROL
                points = [
                    (pos[0], pos[1] - size),
                    (pos[0] - size, pos[1] + size // 2),
                    (pos[0] + size, pos[1] + size // 2)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif enemy['type'] == 'chase':
                color = self.COLOR_ENEMY_CHASE
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, color)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, color)
            elif enemy['type'] == 'random':
                color = self.COLOR_ENEMY_RANDOM
                points = [
                    (pos[0], pos[1] - size),
                    (pos[0] + size, pos[1]),
                    (pos[0], pos[1] + size),
                    (pos[0] - size, pos[1])
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)

        # Player
        if not self.game_over or self.win:
            pos = self.player_pos.astype(int)
            size = int(self.PLAYER_SIZE)
            # Glow effect
            glow_rect = pygame.Rect(pos[0] - size, pos[1] - size, size * 2, size * 2)
            shape_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, self.COLOR_PLAYER_GLOW, (0, 0, size * 2, size * 2), border_radius=4)
            self.screen.blit(shape_surf, glow_rect)
            # Main body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (pos[0] - size/2, pos[1] - size/2, size, size), border_radius=2)
    
    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Fruits to collect
        fruits_left = max(0, self.FRUITS_TO_WIN - self.fruits_collected)
        fruit_text = self.font_ui.render(f"FRUITS: {fruits_left}", True, self.COLOR_UI_TEXT)
        text_rect = fruit_text.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(fruit_text, text_rect)
        
        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_end.render(message, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_collected": self.fruits_collected
        }

    def _get_dist_to_nearest_fruit(self):
        if not self.fruits:
            return self.WIDTH  # Return a large distance if no fruits exist
        
        distances = [np.linalg.norm(self.player_pos - f['pos']) for f in self.fruits]
        return min(distances)

    def _get_dist_to_nearest_enemy(self):
        if not self.enemies:
            return self.WIDTH
        distances = [np.linalg.norm(self.player_pos - e['pos']) for e in self.enemies]
        return min(distances)

    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) if not is_explosion else self.np_random.uniform(2, 7)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'size': self.np_random.uniform(1, 4),
                'lifetime': self.np_random.integers(15, 30),
                'color': color,
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
            p['size'] *= 0.95
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['size'] > 0.5]
    
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly with a graphical interface.
    # To do so, you must unset the dummy video driver.
    # You may need to install a backend like 'pip install pygame[sdl2]'
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup a visible window
    pygame.display.set_caption("Fruit Dodger")
    screen_display = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print(env.user_guide)

    # Game loop
    while not terminated and not truncated:
        # --- Human Controls ---
        movement = 0 # No-op
        action = [0, 0, 0]
        
        # This loop is necessary to handle window closing events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        if terminated:
            break

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Rendering ---
        # The observation is already a rendered frame, just blit it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_display.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    
    # Keep the final screen visible for a moment
    pygame.time.wait(2000)

    env.close()