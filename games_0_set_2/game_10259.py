import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:11:18.993991
# Source Brief: brief_00259.md
# Brief Index: 259
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Synchronize two robotic arms to catch transforming objects falling from the sky.

    The player controls the horizontal movement of two arms at the bottom of the
    screen. The speed of movement can be set to slow, medium, or fast. Objects
    fall from the top, morphing between different shapes. The goal is to catch
    50 objects without letting any hit the ground.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Synchronize two robotic arms to catch transforming objects falling from the sky."
    )
    user_guide = (
        "Use A/D to move the left arm and ←/→ arrows to move the right arm. Hold Shift for slow movement and Space for fast."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GROUND_Y = 360
        self.ARM_WIDTH = 80
        self.ARM_HEIGHT = 10
        self.CATCH_ZONE_HEIGHT = 20

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_LEFT_ARM = (0, 191, 255) # DeepSkyBlue
        self.COLOR_RIGHT_ARM = (255, 140, 0) # DarkOrange
        self.COLOR_OBJECT = (240, 240, 240)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SUCCESS = (50, 205, 50) # LimeGreen
        self.COLOR_FAIL = (220, 20, 60) # Crimson

        # Game parameters
        self.WIN_SCORE = 50
        self.MAX_STEPS = 3000 # Increased for a longer game
        self.ARM_SPEEDS = {'slow': 3, 'medium': 6, 'fast': 10}
        
        self.objects = []
        self.particles = []
        self.stars = []
        
        # Pre-defined shapes (normalized vertices)
        self.SHAPES = {
            'square': [pygame.math.Vector2(x, y) for x, y in [(-1, -1), (1, -1), (1, 1), (-1, 1)]],
            'triangle': [pygame.math.Vector2(x, y) for x, y in [(0, -1.2), (-1.2, 0.8), (1.2, 0.8)]],
            'diamond': [pygame.math.Vector2(x, y) for x, y in [(0, -1.2), (-1, 0), (0, 1.2), (1, 0)]],
            'star': [pygame.math.Vector2(0.0, -1.0), pygame.math.Vector2(0.22, -0.31), pygame.math.Vector2(0.95, -0.31),
                     pygame.math.Vector2(0.36, 0.12), pygame.math.Vector2(0.59, 0.81), pygame.math.Vector2(0.0, 0.38),
                     pygame.math.Vector2(-0.59, 0.81), pygame.math.Vector2(-0.36, 0.12), pygame.math.Vector2(-0.95, -0.31),
                     pygame.math.Vector2(-0.22, -0.31)]
        }
        self.shape_names = list(self.SHAPES.keys())

        # Initialize state variables
        self.reset()

        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Arm states
        self.left_arm_pos = pygame.math.Vector2(self.SCREEN_WIDTH * 0.25, self.GROUND_Y)
        self.right_arm_pos = pygame.math.Vector2(self.SCREEN_WIDTH * 0.75, self.GROUND_Y)

        # Object and particle lists
        self.objects = []
        self.particles = []
        self._spawn_object()

        # Difficulty scaling
        self.current_fall_speed = 2.0
        self.current_morph_rate = 0.01
        
        # Background stars
        if not self.stars:
            for _ in range(100):
                self.stars.append({
                    'pos': pygame.math.Vector2(random.uniform(0, self.SCREEN_WIDTH), random.uniform(0, self.SCREEN_HEIGHT)),
                    'size': random.uniform(0.5, 1.5),
                    'speed': random.uniform(0.1, 0.3)
                })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # 1. Handle Actions and Update Arm Positions
        self._handle_actions(action)
        
        # 2. Update Game Entities
        self._update_objects()
        self._update_particles()
        self._update_stars()

        # 3. Check for Collisions and Misses, Calculate Reward
        reward = self._check_collisions_and_get_reward()
        
        # 4. Check for Termination Conditions
        terminated = self.game_over or self.steps >= self.MAX_STEPS or self.score >= self.WIN_SCORE
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
             if self.score >= self.WIN_SCORE:
                 reward += 100 # Win bonus
                 self._create_text_effect("VICTORY!", self.COLOR_SUCCESS)
             self.game_over = True
        
        # 5. Spawn new object if needed
        if not self.objects and not self.game_over:
            self._spawn_object()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Determine arm speed
        if space_held:
            speed = self.ARM_SPEEDS['fast']
        elif shift_held:
            speed = self.ARM_SPEEDS['slow']
        else:
            speed = self.ARM_SPEEDS['medium']

        # Unpack movement action and update arm positions
        # 1 (Up) -> Left arm Left, 2 (Down) -> Left arm Right
        if movement == 1:
            self.left_arm_pos.x -= speed
        elif movement == 2:
            self.left_arm_pos.x += speed
        
        # 3 (Left) -> Right arm Left, 4 (Right) -> Right arm Right
        if movement == 3:
            self.right_arm_pos.x -= speed
        elif movement == 4:
            self.right_arm_pos.x += speed

        # Clamp arm positions to screen boundaries
        self.left_arm_pos.x = max(self.ARM_WIDTH / 2, min(self.left_arm_pos.x, self.SCREEN_WIDTH - self.ARM_WIDTH / 2))
        self.right_arm_pos.x = max(self.ARM_WIDTH / 2, min(self.right_arm_pos.x, self.SCREEN_WIDTH - self.ARM_WIDTH / 2))

    def _update_objects(self):
        for obj in self.objects:
            obj['pos'].y += self.current_fall_speed
            obj['morph_progress'] = min(1.0, obj['morph_progress'] + self.current_morph_rate)
            if obj['morph_progress'] >= 1.0:
                obj['from_shape'] = obj['to_shape']
                obj['to_shape'] = random.choice(self.shape_names)
                obj['morph_progress'] = 0.0

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            if p.get('shrink', False):
                p['size'] = max(0, p['size'] - 0.1)

    def _update_stars(self):
        for star in self.stars:
            star['pos'].y += star['speed']
            if star['pos'].y > self.SCREEN_HEIGHT:
                star['pos'].y = 0
                star['pos'].x = random.uniform(0, self.SCREEN_WIDTH)

    def _check_collisions_and_get_reward(self):
        reward = 0
        objects_to_remove = []

        for i, obj in enumerate(self.objects):
            # Proximity reward
            proximity_window_y = 150
            left_arm_rect = self._get_arm_rect(self.left_arm_pos)
            right_arm_rect = self._get_arm_rect(self.right_arm_pos)

            if left_arm_rect.left < obj['pos'].x < left_arm_rect.right and self.GROUND_Y - proximity_window_y < obj['pos'].y < self.GROUND_Y:
                reward += 0.1
            if right_arm_rect.left < obj['pos'].x < right_arm_rect.right and self.GROUND_Y - proximity_window_y < obj['pos'].y < self.GROUND_Y:
                reward += 0.1

            # Catch condition
            caught = False
            if self._get_arm_catch_zone(self.left_arm_pos).collidepoint(obj['pos']):
                # sound: catch_left.wav
                self._create_particles(obj['pos'], self.COLOR_SUCCESS)
                self.score += 1
                reward += 10
                caught = True
            elif self._get_arm_catch_zone(self.right_arm_pos).collidepoint(obj['pos']):
                # sound: catch_right.wav
                self._create_particles(obj['pos'], self.COLOR_SUCCESS)
                self.score += 1
                reward += 10
                caught = True
            
            if caught:
                objects_to_remove.append(i)
                # Difficulty scaling
                if self.score % 10 == 0 and self.score > 0:
                    self.current_fall_speed += 0.05
                    self.current_morph_rate += 0.002 # Slower increase
                    self._create_text_effect("SPEED UP!", self.COLOR_RIGHT_ARM)

            # Miss condition
            elif obj['pos'].y > self.GROUND_Y:
                # sound: miss.wav
                self._create_particles(obj['pos'], self.COLOR_FAIL, 50, is_explosion=True)
                self._create_text_effect("MISS!", self.COLOR_FAIL)
                reward -= 100
                self.game_over = True
                objects_to_remove.append(i)
        
        # Remove caught/missed objects
        for i in sorted(objects_to_remove, reverse=True):
            del self.objects[i]
            
        return reward

    def _spawn_object(self):
        self.objects.append({
            'pos': pygame.math.Vector2(random.randint(50, self.SCREEN_WIDTH - 50), -20),
            'size': random.randint(15, 25),
            'from_shape': random.choice(self.shape_names),
            'to_shape': random.choice(self.shape_names),
            'morph_progress': 0.0
        })

    def _create_particles(self, pos, color, count=30, is_explosion=False):
        for _ in range(count):
            if is_explosion:
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 6)
                vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            else: # Upward burst
                vel = pygame.math.Vector2(random.uniform(-1.5, 1.5), random.uniform(-3, -1))
            
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': color,
                'size': random.uniform(1, 4),
                'shrink': True
            })

    def _create_text_effect(self, text, color):
        self.particles.append({
            'pos': pygame.math.Vector2(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2),
            'vel': pygame.math.Vector2(0, -1),
            'lifespan': 60,
            'color': color,
            'text': text,
            'font': self.font
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_background()
        self._render_arms()
        self._render_objects()
        self._render_particles()
        self._render_ui()

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, (200, 200, 255), star['pos'], star['size'])
        pygame.draw.line(self.screen, (100, 100, 120), (0, self.GROUND_Y), (self.SCREEN_WIDTH, self.GROUND_Y), 2)

    def _render_arms(self):
        # Left arm
        left_rect = self._get_arm_rect(self.left_arm_pos)
        self._draw_glowing_rect(self.screen, left_rect, self.COLOR_LEFT_ARM, 15)
        
        # Right arm
        right_rect = self._get_arm_rect(self.right_arm_pos)
        self._draw_glowing_rect(self.screen, right_rect, self.COLOR_RIGHT_ARM, 15)

    def _render_objects(self):
        for obj in self.objects:
            from_verts = self.SHAPES[obj['from_shape']]
            to_verts = self.SHAPES[obj['to_shape']]
            t = obj['morph_progress']
            
            points = []
            for i in range(len(from_verts)):
                # Ensure vertex lists match length, wrap around if needed
                v1 = from_verts[i]
                v2 = to_verts[i % len(to_verts)]
                
                interpolated_v = v1.lerp(v2, t)
                screen_point = obj['pos'] + interpolated_v * obj['size']
                points.append((int(screen_point.x), int(screen_point.y)))
            
            self._draw_glowing_polygon(self.screen, points, self.COLOR_OBJECT, 10)

    def _render_particles(self):
        for p in self.particles:
            if 'text' in p: # Render text particles
                alpha = int(255 * (p['lifespan'] / 60))
                text_surf = p['font'].render(p['text'], True, p['color'])
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=p['pos'])
                self.screen.blit(text_surf, text_rect)
            else: # Render graphical particles
                alpha = int(255 * (p['lifespan'] / 40))
                color = (*p['color'], alpha)
                temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
                self.screen.blit(temp_surf, p['pos'] - pygame.math.Vector2(p['size'], p['size']), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        score_text = f"CAUGHT: {self.score} / {self.WIN_SCORE}"
        text_surf = self.font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        steps_text = f"STEPS: {self.steps} / {self.MAX_STEPS}"
        steps_surf = self.small_font.render(steps_text, True, self.COLOR_TEXT)
        steps_rect = steps_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_surf, steps_rect)
    
    def _draw_glowing_rect(self, surface, rect, color, glow_size):
        for i in range(glow_size, 0, -2):
            alpha = 40 * (1 - i / glow_size)
            glow_color = (*color, alpha)
            glow_rect = rect.inflate(i, i)
            
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(s, glow_color, s.get_rect(), border_radius=5)
            surface.blit(s, glow_rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)
        
        pygame.draw.rect(surface, color, rect, border_radius=3)

    def _draw_glowing_polygon(self, surface, points, color, glow_size):
        if len(points) < 3: return
        
        # Draw glow layers
        for i in range(glow_size, 0, -2):
            alpha = 60 * (1 - i / glow_size)
            glow_color = (*color, alpha)
            
            # Simple offset-based glow
            offset_points = [(p[0], p[1] + i/4) for p in points]
            pygame.gfxdraw.aapolygon(surface, offset_points, glow_color)
            
        # Draw main polygon
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

    def _get_arm_rect(self, pos):
        return pygame.Rect(pos.x - self.ARM_WIDTH / 2, pos.y - self.ARM_HEIGHT / 2, self.ARM_WIDTH, self.ARM_HEIGHT)
    
    def _get_arm_catch_zone(self, pos):
        return pygame.Rect(pos.x - self.ARM_WIDTH / 2, pos.y - self.CATCH_ZONE_HEIGHT, self.ARM_WIDTH, self.CATCH_ZONE_HEIGHT)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fall_speed": self.current_fall_speed,
            "morph_rate": self.current_morph_rate
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # Manual play loop
    # Un-dummy the video driver to see the game
    os.environ["SDL_VIDEODRIVER"] = "x11"
    pygame.display.init()
    pygame.display.set_caption("Robo-Catcher")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("----------------------\n")

    while not done:
        # Action mapping for human play
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        # Left arm movement
        if keys[pygame.K_a]:
            movement = 1 # Corresponds to "Up" for left arm left
        elif keys[pygame.K_d]:
            movement = 2 # Corresponds to "Down" for left arm right
        # Right arm movement
        elif keys[pygame.K_LEFT]:
            movement = 3 # Corresponds to "Left" for right arm left
        elif keys[pygame.K_RIGHT]:
            movement = 4 # Corresponds to "Right" for right arm right

        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(60) # Run at 60 FPS for smoother visuals

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()