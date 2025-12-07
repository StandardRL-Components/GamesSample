
# Generated: 2025-08-28T05:36:46.302512
# Source Brief: brief_02661.md
# Brief Index: 2661

        
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


# Helper class for falling objects (fruits and bombs)
class FallingObject:
    def __init__(self, is_bomb, screen_width, fall_speed, seed=None):
        self.rng = np.random.default_rng(seed)
        self.is_bomb = is_bomb
        self.radius = 18 if is_bomb else 15
        self.pos = pygame.Vector2(self.rng.uniform(self.radius, screen_width - self.radius), -self.radius)
        
        angle_rad = self.rng.uniform(math.pi * 0.4, math.pi * 0.6) # Fall mostly downwards
        self.velocity = pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * fall_speed
        
        if is_bomb:
            self.color = (20, 20, 20)
        else:
            # Bright fruit colors
            self.color = random.choice([(255, 60, 60), (60, 255, 60), (255, 255, 60), (255, 120, 0)])
            self.shine_color = (255, 255, 255, 150)

    def update(self):
        self.pos += self.velocity

    def draw(self, surface):
        # Main body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)

        if not self.is_bomb:
            # Fruit shine
            shine_pos = (int(self.pos.x - self.radius * 0.3), int(self.pos.y - self.radius * 0.3))
            pygame.gfxdraw.filled_circle(surface, shine_pos[0], shine_pos[1], int(self.radius * 0.3), self.shine_color)
        else:
            # Bomb fuse
            fuse_base_rect = pygame.Rect(self.pos.x - 3, self.pos.y - self.radius - 5, 6, 5)
            pygame.draw.rect(surface, (100, 100, 100), fuse_base_rect)
            
            fuse_end_pos = (self.pos.x + 4, self.pos.y - self.radius - 8)
            pygame.draw.line(surface, (200, 200, 150), (self.pos.x, self.pos.y - self.radius - 5), fuse_end_pos, 2)
            pygame.gfxdraw.filled_circle(surface, int(fuse_end_pos[0]), int(fuse_end_pos[1]), 3, (255, 180, 0))


# Helper class for particles
class Particle:
    def __init__(self, pos, color, seed=None):
        self.rng = np.random.default_rng(seed)
        self.pos = pygame.Vector2(pos)
        self.color = color
        angle = self.rng.uniform(0, 2 * math.pi)
        speed = self.rng.uniform(1, 5)
        self.velocity = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.lifespan = self.rng.integers(20, 40)
        self.radius = self.rng.integers(3, 7)

    def update(self):
        self.pos += self.velocity
        self.velocity *= 0.95 # friction
        self.lifespan -= 1
        self.radius -= 0.1

    def draw(self, surface):
        if self.lifespan > 0 and self.radius > 0:
            alpha = max(0, min(255, int(255 * (self.lifespan / 30))))
            color_with_alpha = self.color + (alpha,)
            
            # Create a temporary surface for alpha blending
            temp_surf = pygame.Surface((int(self.radius) * 2, int(self.radius) * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (int(self.radius), int(self.radius)), int(self.radius))
            surface.blit(temp_surf, (self.pos.x - self.radius, self.pos.y - self.radius))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the slicer. Slice the fruit and avoid the bombs!"
    )

    game_description = (
        "A fast-paced arcade game where you slice falling fruit while dodging bombs. "
        "Slice all the fruit to win, but hitting three bombs ends the game."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400

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
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG_TOP = (15, 25, 40)
        self.COLOR_BG_BOTTOM = (5, 10, 20)
        self.COLOR_SLICER = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        
        # Game parameters
        self.SLICER_SPEED = 10
        self.SLICER_TRAIL_LENGTH = 10
        self.MAX_OBJECTS = 8
        self.BOMB_CHANCE = 0.25
        self.FRUITS_TO_WIN = 20
        self.MAX_BOMBS_HIT = 3
        self.MAX_STEPS = 2000
        self.INITIAL_FALL_SPEED = 2.0
        self.DIFFICULTY_INCREMENT = 0.2

        # State variables will be initialized in reset()
        self.slicer_pos = None
        self.slicer_trail = None
        self.falling_objects = None
        self.particles = None
        self.score = None
        self.steps = None
        self.bombs_hit = None
        self.fruits_sliced_total = None
        self.current_fall_speed = None
        self.screen_shake = 0
        self.seed_value = None

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed_value = seed
        
        # Initialize game state
        self.slicer_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.slicer_trail = []
        self.falling_objects = []
        self.particles = []

        self.steps = 0
        self.score = 0
        self.bombs_hit = 0
        self.fruits_sliced_total = 0
        self.current_fall_speed = self.INITIAL_FALL_SPEED
        self.screen_shake = 0

        # Populate initial objects
        for _ in range(self.MAX_OBJECTS // 2):
            self._spawn_object()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        
        reward = -1.0 # Time penalty
        
        # 1. Handle player input
        self._handle_input(movement)
        
        # 2. Update game state
        self.slicer_trail.append(self.slicer_pos.copy())
        if len(self.slicer_trail) > self.SLICER_TRAIL_LENGTH:
            self.slicer_trail.pop(0)

        # Update particles
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
        # Update falling objects and check for slicing/misses
        new_falling_objects = []
        for obj in self.falling_objects:
            obj.update()
            
            # Check for slice
            dist = self.slicer_pos.distance_to(obj.pos)
            if dist < obj.radius + 5: # +5 for a more generous slice hitbox
                if obj.is_bomb:
                    # Sliced a bomb
                    self.bombs_hit += 1
                    reward -= 50
                    self.score -= 50
                    self._create_explosion_particles(obj.pos)
                    self.screen_shake = 15
                    # Sound: Explosion
                else:
                    # Sliced a fruit
                    self.fruits_sliced_total += 1
                    reward += 1
                    self.score += 10
                    self._create_slice_particles(obj.pos, obj.color)
                    # Sound: Fruit slice
                    if self.fruits_sliced_total > 0 and self.fruits_sliced_total % 10 == 0:
                        self.current_fall_speed += self.DIFFICULTY_INCREMENT
                # Object is removed (not added to new list)
            elif obj.pos.y > self.HEIGHT + obj.radius or obj.pos.x < -obj.radius or obj.pos.x > self.WIDTH + obj.radius:
                # Object went off-screen
                if not obj.is_bomb:
                    reward -= 5 # Penalty for missing fruit
                # Object is removed
            else:
                new_falling_objects.append(obj)
        self.falling_objects = new_falling_objects

        # 3. Spawn new objects
        if len(self.falling_objects) < self.MAX_OBJECTS:
            if self.np_random.random() < 0.1: # Chance to spawn each frame
                self._spawn_object()

        # 4. Update step counter and check for termination
        self.steps += 1
        terminated = False
        if self.bombs_hit >= self.MAX_BOMBS_HIT:
            terminated = True
        elif self.fruits_sliced_total >= self.FRUITS_TO_WIN:
            reward += 100 # Victory bonus
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement):
        if movement == 1: # Up
            self.slicer_pos.y -= self.SLICER_SPEED
        elif movement == 2: # Down
            self.slicer_pos.y += self.SLICER_SPEED
        elif movement == 3: # Left
            self.slicer_pos.x -= self.SLICER_SPEED
        elif movement == 4: # Right
            self.slicer_pos.x += self.SLICER_SPEED
        
        # Clamp slicer position to screen bounds
        self.slicer_pos.x = max(0, min(self.WIDTH, self.slicer_pos.x))
        self.slicer_pos.y = max(0, min(self.HEIGHT, self.slicer_pos.y))

    def _spawn_object(self):
        is_bomb = self.np_random.random() < self.BOMB_CHANCE
        obj = FallingObject(is_bomb, self.WIDTH, self.current_fall_speed, self.seed_value)
        self.falling_objects.append(obj)

    def _create_slice_particles(self, pos, color):
        # Sound: Squish
        for _ in range(20):
            self.particles.append(Particle(pos, color, self.seed_value))
    
    def _create_explosion_particles(self, pos):
        # Sound: Boom
        for _ in range(40):
            color = random.choice([(255, 180, 0), (255, 80, 0), (100, 100, 100)])
            self.particles.append(Particle(pos, color, self.seed_value))

    def _get_observation(self):
        # Create a temporary surface to apply screen shake
        render_surface = self.screen.copy()

        # Render background gradient
        for y in range(self.HEIGHT):
            interp = y / self.HEIGHT
            color = (
                self.COLOR_BG_TOP[0] * (1 - interp) + self.COLOR_BG_BOTTOM[0] * interp,
                self.COLOR_BG_TOP[1] * (1 - interp) + self.COLOR_BG_BOTTOM[1] * interp,
                self.COLOR_BG_TOP[2] * (1 - interp) + self.COLOR_BG_BOTTOM[2] * interp,
            )
            pygame.draw.line(render_surface, color, (0, y), (self.WIDTH, y))

        # Render particles
        for p in self.particles:
            p.draw(render_surface)

        # Render falling objects
        for obj in self.falling_objects:
            obj.draw(render_surface)
            
        # Render slicer trail
        if len(self.slicer_trail) > 1:
            points = self.slicer_trail
            for i in range(len(points) - 1):
                alpha = int(255 * ((i + 1) / len(points)))
                color = (*self.COLOR_SLICER, alpha)
                width = max(1, int(10 * ((i + 1) / len(points))))
                
                # Use a temporary surface for alpha-blended lines
                line_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(line_surf, color, points[i], points[i+1], width)
                render_surface.blit(line_surf, (0,0))

        # Apply screen shake
        offset = (0, 0)
        if self.screen_shake > 0:
            offset = (self.np_random.integers(-self.screen_shake, self.screen_shake + 1), 
                      self.np_random.integers(-self.screen_shake, self.screen_shake + 1))
            self.screen_shake -= 1
        self.screen.blit(render_surface, offset)

        # Render UI
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_ui(self):
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Fruit Sliced
        fruit_text = self.font_small.render(f"Fruit: {self.fruits_sliced_total}/{self.FRUITS_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(fruit_text, (10, 45))

        # Bombs Hit
        bomb_display_text = "Bombs:"
        text_surf = self.font.render(bomb_display_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.WIDTH - text_surf.get_width() - 120, 10))
        for i in range(self.MAX_BOMBS_HIT):
            color = (100, 20, 20) if i < self.bombs_hit else (50, 50, 50)
            pos = (self.WIDTH - 110 + i * 35, 27)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, (20,20,20))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fruits_sliced": self.fruits_sliced_total,
            "bombs_hit": self.bombs_hit
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
        
        # Test observation space (initial from reset)
        initial_obs, _ = self.reset()
        assert initial_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {initial_obs.shape}"
        assert initial_obs.dtype == np.uint8
        
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Fruit Slicer")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop
    running = True
    while running:
        # Action defaults to no-op
        movement = 0 
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_r]: # Press R to reset
            obs, info = env.reset()
            terminated = False

        if not terminated:
            action = [movement, 0, 0] # Space and Shift are not used
            obs, reward, terminated, truncated, info = env.step(action)
            
            # The observation from the env is what we render
            # It needs to be transposed back for pygame's screen.blit
            frame = np.transpose(obs, (1, 0, 2))
            frame_surface = pygame.surfarray.make_surface(frame)
            screen.blit(frame_surface, (0, 0))

            if terminated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    env.close()