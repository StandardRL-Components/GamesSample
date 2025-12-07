
# Generated: 2025-08-27T21:34:59.546651
# Source Brief: brief_02836.md
# Brief Index: 2836

        
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
        "Controls: Use arrow keys (↑↓←→) to move your character. "
        "Catch the falling fruits to score points and avoid the red bombs!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced arcade game where you must collect as many falling fruits as "
        "possible in 60 seconds. Avoid the bombs, as hitting one will end the game immediately."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60
        self.GAME_DURATION_SECONDS = 60
        
        # Player settings
        self.PLAYER_RADIUS = 15
        self.PLAYER_SPEED = 6
        
        # Entity settings
        self.FRUIT_RADIUS = 10
        self.BOMB_RADIUS = 12
        self.ENTITY_SPEED_MIN = 2
        self.ENTITY_SPEED_MAX = 4
        
        # Colors
        self.COLOR_BG_TOP = (20, 30, 50)
        self.COLOR_BG_BOTTOM = (40, 60, 90)
        self.COLOR_PLAYER = (0, 255, 255) # Cyan
        self.COLOR_PLAYER_GLOW = (0, 150, 150)
        self.FRUIT_COLORS = {
            "apple": (255, 60, 60),
            "banana": (255, 255, 80),
            "grape": (180, 80, 255),
        }
        self.COLOR_BOMB = (200, 0, 0)
        self.COLOR_BOMB_FLASH = (255, 100, 100)
        self.COLOR_TEXT = (255, 255, 255)
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 24)

        # State variables (will be initialized in reset)
        self.player_pos = None
        self.fruits = None
        self.bombs = None
        self.particles = None
        self.score = None
        self.steps = None
        self.time_remaining = None
        self.game_over = None
        self.fruit_spawn_timer = None
        self.bomb_spawn_timer = None
        self.np_random = None

        # Initialize state
        self.reset()

        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.player_pos = [self.WIDTH // 2, self.HEIGHT - self.PLAYER_RADIUS - 30]
        self.fruits = []
        self.bombs = []
        self.particles = []
        
        self.score = 0
        self.steps = 0
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.game_over = False
        
        self._update_spawn_intervals()
        self.fruit_spawn_timer = self.np_random.integers(0, self.fruit_spawn_interval)
        self.bomb_spawn_timer = self.np_random.integers(0, self.bomb_spawn_interval)
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Update Game State ---
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # Unpack factorized action
        movement = action[0]
        
        self._handle_player_movement(movement)
        self._update_entities()
        self._handle_spawning()
        
        # --- Collision Detection and Rewards ---
        reward += self._handle_collisions()
        
        # --- Termination Check ---
        terminated = self.game_over or self.time_remaining <= 0 or self.score >= 50
        
        if not self.game_over and self.score >= 50:
            reward += 100 # Victory bonus
            self.game_over = True
        
        if self.time_remaining <= 0 and not self.game_over:
            self.game_over = True # Time's up

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_player_movement(self, movement):
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
            
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_entities(self):
        # Update fruits and bombs
        for entity in self.fruits + self.bombs:
            entity['pos'][1] += entity['speed']
        
        # Remove off-screen entities
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.HEIGHT + self.FRUIT_RADIUS]
        self.bombs = [b for b in self.bombs if b['pos'][1] < self.HEIGHT + self.BOMB_RADIUS]
        
        # Update particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def _update_spawn_intervals(self):
        time_elapsed = self.GAME_DURATION_SECONDS - self.time_remaining
        difficulty_tier = math.floor(time_elapsed / 10)
        base_spawn_rate = 0.5  # spawns/sec
        rate_increase = 0.2
        
        current_spawn_rate = min(2.0, base_spawn_rate + difficulty_tier * rate_increase)
        spawn_interval_seconds = 1.0 / current_spawn_rate
        self.fruit_spawn_interval = int(spawn_interval_seconds * self.FPS)
        self.bomb_spawn_interval = int(spawn_interval_seconds * self.FPS)

    def _handle_spawning(self):
        self._update_spawn_intervals()
        
        self.fruit_spawn_timer -= 1
        if self.fruit_spawn_timer <= 0:
            self._spawn_fruit()
            self.fruit_spawn_timer = self.fruit_spawn_interval

        self.bomb_spawn_timer -= 1
        if self.bomb_spawn_timer <= 0:
            self._spawn_bomb()
            self.bomb_spawn_timer = self.bomb_spawn_interval
            
    def _spawn_fruit(self):
        fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
        self.fruits.append({
            'pos': [self.np_random.integers(self.FRUIT_RADIUS, self.WIDTH - self.FRUIT_RADIUS), -self.FRUIT_RADIUS],
            'speed': self.np_random.uniform(self.ENTITY_SPEED_MIN, self.ENTITY_SPEED_MAX),
            'type': fruit_type,
            'color': self.FRUIT_COLORS[fruit_type]
        })

    def _spawn_bomb(self):
        self.bombs.append({
            'pos': [self.np_random.integers(self.BOMB_RADIUS, self.WIDTH - self.BOMB_RADIUS), -self.BOMB_RADIUS],
            'speed': self.np_random.uniform(self.ENTITY_SPEED_MIN, self.ENTITY_SPEED_MAX)
        })

    def _handle_collisions(self):
        reward = 0
        player_x, player_y = self.player_pos

        # Fruit collisions
        for fruit in self.fruits[:]:
            dist = math.hypot(player_x - fruit['pos'][0], player_y - fruit['pos'][1])
            if dist < self.PLAYER_RADIUS + self.FRUIT_RADIUS:
                self.score += 1
                reward += 1.0 # Base reward for fruit
                # Check for near-bomb bonus
                for bomb in self.bombs:
                    if math.hypot(fruit['pos'][0] - bomb['pos'][0], fruit['pos'][1] - bomb['pos'][1]) < 50:
                        reward += 10.0
                        break # Only one bonus per fruit
                self._create_particles(fruit['pos'], fruit['color'], 20)
                # sfx: fruit_collect.wav
                self.fruits.remove(fruit)

        # Bomb collisions
        for bomb in self.bombs:
            dist = math.hypot(player_x - bomb['pos'][0], player_y - bomb['pos'][1])
            if dist < self.PLAYER_RADIUS + self.BOMB_RADIUS:
                self.game_over = True
                reward = -5.0 # Penalty for hitting a bomb
                self._create_particles(self.player_pos, self.COLOR_BOMB, 50, is_explosion=True)
                # sfx: explosion.wav
                break
        return reward

    def _create_particles(self, pos, color, count, is_explosion=False):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5 if is_explosion else 3)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(1, 4)
            
            if is_explosion:
                # Explosion particles are fiery colors
                p_color = random.choice([(255,0,0), (255,165,0), (255,255,0), (100,100,100)])
            else:
                p_color = color

            self.particles.append({
                'pos': list(pos), 
                'vel': vel, 
                'lifetime': lifetime, 
                'max_lifetime': lifetime,
                'radius': radius,
                'color': p_color
            })
    
    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
        }

    def _render_background(self):
        # Draw a vertical gradient
        for y in range(self.HEIGHT):
            ratio = y / self.HEIGHT
            color = [
                int(self.COLOR_BG_TOP[i] * (1 - ratio) + self.COLOR_BG_BOTTOM[i] * ratio)
                for i in range(3)
            ]
            pygame.draw.line(self.screen, color, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['lifetime'] / p['max_lifetime']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

        # Render bombs
        for bomb in self.bombs:
            pos = (int(bomb['pos'][0]), int(bomb['pos'][1]))
            # Flashing effect
            flash = (self.steps // 6) % 2 == 0
            color = self.COLOR_BOMB_FLASH if flash else self.COLOR_BOMB
            radius = self.BOMB_RADIUS + (2 if flash else 0)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)

        # Render fruits
        for fruit in self.fruits:
            pos = (int(fruit['pos'][0]), int(fruit['pos'][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.FRUIT_RADIUS, fruit['color'])
            
        # Render player
        if not (self.game_over and self.time_remaining > 0): # Hide player on bomb collision
            player_pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
            # Glow effect
            glow_radius = self.PLAYER_RADIUS + 5 + int(3 * math.sin(self.steps * 0.1))
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (player_pos_int[0] - glow_radius, player_pos_int[1] - glow_radius))

            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        # Render score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))
        
        # Render timer
        time_str = f"TIME: {max(0, math.ceil(self.time_remaining)):02d}"
        timer_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Render game over message
        if self.game_over:
            if self.score >= 50:
                msg = "YOU WIN!"
            elif self.time_remaining <= 0:
                msg = "TIME'S UP!"
            else:
                msg = "GAME OVER"
            
            over_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            over_rect = over_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 20))
            pygame.draw.rect(self.screen, (0,0,0,150), over_rect.inflate(20,20))
            self.screen.blit(over_text, over_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

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

# Example usage to test the environment
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This requires setting up a display window
    pygame.display.set_caption("Fruit Catcher")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Convert pygame key presses to actions
    action = env.action_space.sample()
    action[0] = 0 # No-op
    action[1] = 0 # Space released
    action[2] = 0 # Shift released

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        else:
            action[0] = 0 # No movement

        # Other actions (unused in this game)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated:
            print(f"Game Over! Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()