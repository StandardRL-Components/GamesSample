
# Generated: 2025-08-28T02:12:55.832249
# Source Brief: brief_04372.md
# Brief Index: 4372

        
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

    # Short, user-facing control string
    user_guide = "Controls: Use ← and → to move the basket. Catch fruit, avoid bombs!"

    # Short, user-facing description of the game
    game_description = "Catch falling fruit and avoid bombs in this fast-paced isometric arcade game to reach a target score."

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG_TOP = (5, 20, 40)
    COLOR_BG_BOTTOM = (20, 50, 90)
    COLOR_BASKET_BODY = (160, 82, 45)
    COLOR_BASKET_RIM = (139, 69, 19)
    COLOR_BASKET_SHADOW = (0, 0, 0, 50)
    COLOR_BOMB = (30, 30, 30)
    COLOR_BOMB_FUSE = (150, 150, 120)
    COLOR_BOMB_SPARK = (255, 200, 0)
    COLOR_TEXT = (255, 255, 255)
    COLOR_TEXT_SHADOW = (0, 0, 0, 128)
    FRUIT_COLORS = {
        "apple": ((220, 40, 40), (255, 80, 80)),
        "lemon": ((255, 230, 20), (255, 255, 100)),
        "grape": ((120, 40, 180), (160, 80, 220)),
    }
    PARTICLE_COLORS = {
        "apple": (220, 40, 40),
        "lemon": (255, 230, 20),
        "grape": (120, 40, 180),
        "explosion": ((255, 100, 0), (255, 200, 0), (255, 255, 255))
    }
    
    # Game Parameters
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    TARGET_SCORE = 100
    MAX_BOMBS = 3
    MAX_STEPS = 1000
    BASKET_WIDTH_BOTTOM = 60
    BASKET_WIDTH_TOP = 80
    BASKET_HEIGHT = 20
    BASKET_Y = SCREEN_HEIGHT - 40
    BASKET_SPEED = 10.0
    INITIAL_FALL_SPEED = 1.5
    SPAWN_RATE = 30 # Frames between spawns
    DIFFICULTY_INTERVAL = 100 # Steps to increase difficulty
    SPEED_INCREASE = 0.05
    BOMB_PROBABILITY = 0.25

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        # State variables are initialized in reset()
        self.basket_x = 0
        self.score = 0
        self.bombs_caught = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0
        self.spawn_timer = 0
        self.falling_objects = []
        self.particles = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.basket_x = self.SCREEN_WIDTH / 2
        self.score = 0
        self.bombs_caught = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = self.INITIAL_FALL_SPEED
        self.spawn_timer = self.SPAWN_RATE
        self.falling_objects = []
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_objects()
        
        # Collision detection and handling
        reward += self._handle_collisions()
        
        self._update_particles()
        
        self.steps += 1
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.fall_speed += self.SPEED_INCREASE

        # --- Check Termination Conditions ---
        terminated = False
        if self.score >= self.TARGET_SCORE:
            reward += 50 # Win bonus
            terminated = True
            self.game_over = True
        elif self.bombs_caught >= self.MAX_BOMBS:
            reward -= 50 # Bomb loss penalty
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

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 3:  # Left
            self.basket_x -= self.BASKET_SPEED
        elif movement == 4:  # Right
            self.basket_x += self.BASKET_SPEED
            
        # Clamp basket position to screen bounds
        half_basket_width = self.BASKET_WIDTH_TOP / 2
        self.basket_x = max(half_basket_width, min(self.SCREEN_WIDTH - half_basket_width, self.basket_x))

    def _update_objects(self):
        # Spawn new objects
        self.spawn_timer -= 1
        if self.spawn_timer <= 0:
            self._spawn_object()
            self.spawn_timer = self.SPAWN_RATE

        # Move existing objects and remove off-screen ones
        new_objects = []
        for obj in self.falling_objects:
            obj['y'] += self.fall_speed
            obj['x'] += obj['drift']
            if obj['y'] < self.SCREEN_HEIGHT + 20:
                new_objects.append(obj)
        self.falling_objects = new_objects

    def _spawn_object(self):
        x_pos = random.uniform(20, self.SCREEN_WIDTH - 20)
        is_bomb = self.np_random.random() < self.BOMB_PROBABILITY
        
        if is_bomb:
            obj = {
                "type": "bomb",
                "x": x_pos, "y": -20,
                "size": 12,
                "drift": random.uniform(-0.2, 0.2),
            }
        else:
            fruit_type = self.np_random.choice(list(self.FRUIT_COLORS.keys()))
            obj = {
                "type": fruit_type,
                "x": x_pos, "y": -20,
                "size": 10,
                "drift": random.uniform(-0.1, 0.1),
            }
        self.falling_objects.append(obj)

    def _handle_collisions(self):
        reward = 0
        remaining_objects = []
        basket_left = self.basket_x - self.BASKET_WIDTH_TOP / 2
        basket_right = self.basket_x + self.BASKET_WIDTH_TOP / 2
        basket_top = self.BASKET_Y - self.BASKET_HEIGHT / 2
        
        for obj in self.falling_objects:
            is_caught = False
            if basket_top < obj['y'] < self.BASKET_Y + 5:
                if basket_left < obj['x'] < basket_right:
                    is_caught = True
                    if obj['type'] == 'bomb':
                        # SFX: Explosion
                        self.bombs_caught += 1
                        reward -= 5
                        self._create_explosion(obj['x'], obj['y'])
                    else:
                        # SFX: Fruit catch
                        self.score += 1
                        reward += 1
                        self._create_splash(obj['x'], obj['y'], obj['type'])
            
            if not is_caught:
                remaining_objects.append(obj)
                
        self.falling_objects = remaining_objects
        return reward

    def _create_splash(self, x, y, fruit_type):
        color = self.PARTICLE_COLORS[fruit_type]
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "size": random.uniform(2, 5),
                "color": color,
                "life": 20,
            })

    def _create_explosion(self, x, y):
        for _ in range(40):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 7)
            color = random.choice(self.PARTICLE_COLORS["explosion"])
            self.particles.append({
                "x": x, "y": y,
                "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
                "size": random.uniform(3, 8),
                "color": color,
                "life": random.randint(20, 40),
            })

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1
            p['size'] *= 0.95
            if p['life'] > 0 and p['size'] > 0.5:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        # Clear screen with background gradient
        self._render_background()
        
        # Render game elements
        self._render_basket_shadow()
        self._render_objects()
        self._render_particles()
        self._render_basket()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bombs_caught": self.bombs_caught,
        }
        
    def _render_background(self):
        for y in range(self.SCREEN_HEIGHT):
            interp = y / self.SCREEN_HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(self.screen, color, (0, y), (self.SCREEN_WIDTH, y))

    def _render_basket_shadow(self):
        shadow_surface = pygame.Surface((self.BASKET_WIDTH_TOP + 20, 20), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, self.COLOR_BASKET_SHADOW, shadow_surface.get_rect())
        self.screen.blit(shadow_surface, (int(self.basket_x - (self.BASKET_WIDTH_TOP/2) - 10), int(self.BASKET_Y + self.BASKET_HEIGHT - 5)))

    def _render_basket(self):
        x, y = int(self.basket_x), int(self.BASKET_Y)
        w_bot, w_top = self.BASKET_WIDTH_BOTTOM, self.BASKET_WIDTH_TOP
        h = self.BASKET_HEIGHT

        # Main body
        points = [
            (x - w_bot / 2, y + h), (x + w_bot / 2, y + h),
            (x + w_top / 2, y), (x - w_top / 2, y)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_BASKET_RIM)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_BASKET_BODY)
        
        # Rim
        pygame.draw.line(self.screen, self.COLOR_BASKET_RIM, (x - w_top / 2, y), (x + w_top / 2, y), 3)

    def _render_objects(self):
        for obj in self.falling_objects:
            if obj['type'] == 'bomb':
                self._render_bomb(obj)
            else:
                self._render_fruit(obj)
                
    def _render_fruit(self, obj):
        x, y, size = int(obj['x']), int(obj['y']), int(obj['size'])
        main_color, highlight_color = self.FRUIT_COLORS[obj['type']]
        
        # Main fruit body with antialiasing
        pygame.gfxdraw.aacircle(self.screen, x, y, size, main_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, size, main_color)

        # Highlight
        pygame.gfxdraw.aacircle(self.screen, x-int(size*0.2), y-int(size*0.2), int(size*0.4), highlight_color)
        pygame.gfxdraw.filled_circle(self.screen, x-int(size*0.2), y-int(size*0.2), int(size*0.4), highlight_color)

    def _render_bomb(self, obj):
        x, y, size = int(obj['x']), int(obj['y']), int(obj['size'])
        
        # Bomb body
        pygame.gfxdraw.aacircle(self.screen, x, y, size, self.COLOR_BOMB)
        pygame.gfxdraw.filled_circle(self.screen, x, y, size, self.COLOR_BOMB)
        
        # Fuse
        fuse_path = [
            (x, y - size), (x + 3, y - size - 3), (x, y - size - 6), (x - 3, y - size - 9)
        ]
        pygame.draw.lines(self.screen, self.COLOR_BOMB_FUSE, False, fuse_path, 2)
        
        # Spark
        spark_pos = fuse_path[-1]
        spark_size = 2 if self.steps % 10 < 5 else 3
        pygame.draw.circle(self.screen, self.COLOR_BOMB_SPARK, spark_pos, spark_size)

    def _render_particles(self):
        for p in self.particles:
            size = max(0, int(p['size']))
            if size > 0:
                # Using simple circles for performance; gfxdraw is slower in loops
                pygame.draw.circle(self.screen, p['color'], (int(p['x']), int(p['y'])), size)

    def _render_ui(self):
        # Render Score
        score_text = f"SCORE: {self.score}"
        self._draw_text(score_text, (15, 10), self.font_main)
        
        # Render Bomb Counter (Skulls)
        skull_char = "💀"
        for i in range(self.MAX_BOMBS):
            pos = (self.SCREEN_WIDTH - 30 - i * 30, 10)
            color = self.COLOR_TEXT if i < self.bombs_caught else (100, 100, 100, 128)
            self._draw_text(skull_char, pos, self.font_main, color=color)

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            if self.score >= self.TARGET_SCORE:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            self._draw_text(msg, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20), self.font_main, center=True)
            self._draw_text(f"Final Score: {self.score}", (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20), self.font_small, center=True)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        # Shadow
        text_surf_shadow = font.render(text, True, (0,0,0))
        text_rect_shadow = text_surf_shadow.get_rect()
        if center:
            text_rect_shadow.center = (pos[0] + 2, pos[1] + 2)
        else:
            text_rect_shadow.topleft = (pos[0] + 2, pos[1] + 2)
        self.screen.blit(text_surf_shadow, text_rect_shadow)
        
        # Main text
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # For human play
    pygame.display.set_caption("Fruit Catcher")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = env.action_space.sample()
    action.fill(0)

    while not done:
        # Human controls
        movement = 0 # no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        action = np.array([movement, 0, 0])

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Control the frame rate

    print(f"Game Over! Final Info: {info}")
    env.close()