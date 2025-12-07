import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:34:27.271502
# Source Brief: brief_01141.md
# Brief Index: 1141
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper class for game objects (the falling circles)
class GameObject:
    def __init__(self, pos, radius, color, mass, vel, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.radius = radius
        self.color = color
        self.mass = mass
        self.lifespan = lifespan  # Frames until it is considered "saved" if not lost

    def update(self, gravity):
        self.vel += gravity
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, surface):
        # Draw a soft glow effect
        glow_radius = int(self.radius * 1.5)
        glow_color = self.color + (50,)  # Add alpha
        temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
        surface.blit(temp_surf, (int(self.pos.x - glow_radius), int(self.pos.y - glow_radius)), special_flags=pygame.BLEND_RGBA_ADD)

        # Draw the main circle
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), self.color)

# Helper class for visual particles
class Particle:
    def __init__(self, pos, vel, color, start_size, end_size, lifespan):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.color = color
        self.start_size = start_size
        self.end_size = end_size
        self.lifespan = lifespan
        self.max_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.vel *= 0.95  # Damping
        self.lifespan -= 1

    def draw(self, surface):
        progress = self.lifespan / self.max_lifespan
        current_size = self.start_size * progress + self.end_size * (1 - progress)
        if current_size < 1: return

        # Fade out color
        alpha = int(255 * progress)
        if alpha <= 0: return
        
        rect = pygame.Rect(self.pos.x - current_size / 2, self.pos.y - current_size / 2, current_size, current_size)
        
        # Use a surface with alpha for smooth fading
        temp_surf = pygame.Surface((current_size, current_size), pygame.SRCALPHA)
        temp_surf.fill((*self.color, alpha))
        surface.blit(temp_surf, rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Draw lines with your cursor to deflect falling objects. "
        "Keep them from hitting the floor to score points and survive."
    )
    user_guide = (
        "Controls: Use ↑↓←→ to move the cursor. Hold space to start drawing a line, and hold shift to finish it."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 60)

        # Colors
        self.COLOR_BG = (15, 18, 28)
        self.COLOR_FLOOR = (200, 200, 220)
        self.COLOR_LINE = (255, 255, 255)
        self.COLOR_LINE_DRAWING = (150, 150, 170)
        self.COLOR_CURSOR = (255, 0, 100)
        self.COLOR_UI_TEXT = (230, 230, 240)
        self.OBJECT_COLORS = {
            "light": (255, 80, 80),
            "medium": (80, 255, 80),
            "heavy": (80, 80, 255),
        }
        
        # Game constants
        self.CURSOR_SPEED = 5.0
        self.GRAVITY = pygame.Vector2(0, 0.08)
        self.FLOOR_Y = self.HEIGHT - 10
        self.MAX_STEPS = 1000
        self.WIN_CONDITION = 100
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0.0
        self.saved_objects_count = 0
        self.cursor_pos = pygame.Vector2(0, 0)
        self.is_drawing = False
        self.current_line_start = None
        self.lines = []
        self.objects = []
        self.particles = []
        self.object_spawn_timer = 0
        self.object_fall_speed = 0.0
        self.initial_fall_speed = 0.0
        self.last_space_held = False
        self.last_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0.0
        self.saved_objects_count = 0
        
        self.cursor_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.is_drawing = False
        self.current_line_start = None
        self.lines = []
        self.objects = []
        self.particles = []
        
        self.object_spawn_timer = 0
        self.initial_fall_speed = 0.5
        self.object_fall_speed = self.initial_fall_speed

        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.0
        
        # 1. Handle Input
        self._handle_input(action)

        # 2. Update Game Logic
        reward += self._update_objects()
        self._update_particles()
        self._spawn_objects()
        
        # 3. Update Global State & Check Termination
        self.steps += 1
        terminated = (self.steps >= self.MAX_STEPS) or (self.saved_objects_count >= self.WIN_CONDITION)
        
        if self.saved_objects_count >= self.WIN_CONDITION:
            reward += 100.0 # Goal-oriented reward
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_now, shift_now = action[0], action[1] == 1, action[2] == 1
        
        # Cursor movement
        if movement == 1: self.cursor_pos.y -= self.CURSOR_SPEED
        elif movement == 2: self.cursor_pos.y += self.CURSOR_SPEED
        elif movement == 3: self.cursor_pos.x -= self.CURSOR_SPEED
        elif movement == 4: self.cursor_pos.x += self.CURSOR_SPEED
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.WIDTH)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.HEIGHT)

        # Drawing logic (state change detection)
        if space_now and not self.last_space_held and not self.is_drawing:
            self.is_drawing = True
            self.current_line_start = self.cursor_pos.copy()
            # SFX: StartDraw.wav
        
        if shift_now and not self.last_shift_held and self.is_drawing:
            self.is_drawing = False
            end_pos = self.cursor_pos.copy()
            if self.current_line_start and self.current_line_start.distance_to(end_pos) > 2:
                self.lines.append((self.current_line_start, end_pos))
            self.current_line_start = None
            # SFX: EndDraw.wav

        self.last_space_held = space_now
        self.last_shift_held = shift_now

    def _spawn_objects(self):
        self.object_spawn_timer -= 1
        if self.object_spawn_timer <= 0:
            self.object_spawn_timer = random.randint(30, 60)
            
            x_pos = random.uniform(20, self.WIDTH - 20)
            obj_type = random.choices(["light", "medium", "heavy"], weights=[0.4, 0.4, 0.2], k=1)[0]
            
            if obj_type == "light":
                radius, mass = 8, 1
            elif obj_type == "medium":
                radius, mass = 12, 2
            else: # heavy
                radius, mass = 16, 4
            
            color = self.OBJECT_COLORS[obj_type]
            lifespan = 600 # 20 seconds at 30fps
            self.objects.append(GameObject(
                pos=(x_pos, -radius),
                radius=radius,
                color=color,
                mass=mass,
                vel=(0, self.object_fall_speed),
                lifespan=lifespan
            ))

    def _create_collision_particles(self, pos, color):
        # SFX: Collision.wav
        for _ in range(random.randint(5, 8)):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            lifespan = random.randint(15, 30)
            self.particles.append(Particle(pos, vel, color, 4, 1, lifespan))

    def _update_objects(self):
        step_reward = 0.0
        
        # Apply gravity and move
        for obj in self.objects:
            obj.update(self.GRAVITY)

        # Handle collisions
        # Object-Object
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                obj1 = self.objects[i]
                obj2 = self.objects[j]
                dist_vec = obj1.pos - obj2.pos
                if 0 < dist_vec.length() < obj1.radius + obj2.radius:
                    step_reward += 0.1
                    self._create_collision_particles((obj1.pos + obj2.pos) / 2, obj1.color)
                    
                    # Resolve overlap
                    overlap = (obj1.radius + obj2.radius) - dist_vec.length()
                    normal = dist_vec.normalize()
                    obj1.pos += normal * overlap * 0.5
                    obj2.pos -= normal * overlap * 0.5
                    
                    # Elastic collision physics
                    v1 = obj1.vel
                    v2 = obj2.vel
                    m1 = obj1.mass
                    m2 = obj2.mass
                    x1 = obj1.pos
                    x2 = obj2.pos

                    v1_new = v1 - (2 * m2 / (m1 + m2)) * (v1 - v2).dot(x1 - x2) / (x1 - x2).length_squared() * (x1 - x2)
                    v2_new = v2 - (2 * m1 / (m1 + m2)) * (v2 - v1).dot(x2 - x1) / (x2 - x1).length_squared() * (x2 - x1)
                    
                    obj1.vel = v1_new
                    obj2.vel = v2_new

        # Object-Line and Object-Wall
        for obj in self.objects:
            # Wall collisions
            if obj.pos.x - obj.radius < 0:
                obj.pos.x = obj.radius
                obj.vel.x *= -0.8
            if obj.pos.x + obj.radius > self.WIDTH:
                obj.pos.x = self.WIDTH - obj.radius
                obj.vel.x *= -0.8
            if obj.pos.y - obj.radius < 0 and obj.vel.y < 0: # Top wall bounce only if moving up
                obj.pos.y = obj.radius
                obj.vel.y *= -0.8

            # Line collisions
            for p1, p2 in self.lines:
                # Find closest point on line segment to circle center
                line_vec = p2 - p1
                if line_vec.length() == 0: continue
                
                point_vec = obj.pos - p1
                t = line_vec.dot(point_vec) / line_vec.length_squared()
                t = max(0, min(1, t)) # Clamp to segment
                
                closest_point = p1 + t * line_vec
                dist_vec = obj.pos - closest_point
                
                if 0 < dist_vec.length() < obj.radius:
                    self._create_collision_particles(closest_point, self.COLOR_LINE)
                    
                    # Resolve overlap
                    overlap = obj.radius - dist_vec.length()
                    normal = dist_vec.normalize()
                    obj.pos += normal * overlap
                    
                    # Reflect velocity
                    obj.vel = obj.vel.reflect(normal) * 0.85 # Bounciness

        # Check for saved or lost objects
        objects_to_remove = []
        for i, obj in enumerate(self.objects):
            if obj.pos.y > self.FLOOR_Y + obj.radius:
                objects_to_remove.append(i)
                # SFX: Lost.wav
            elif obj.lifespan <= 0:
                objects_to_remove.append(i)
                self.saved_objects_count += 1
                step_reward += 1.0
                # SFX: Saved.wav
                # Update difficulty
                if self.saved_objects_count > 0 and self.saved_objects_count % 50 == 0:
                    self.object_fall_speed += 0.05

        # Remove objects in reverse order to avoid index errors
        for i in sorted(objects_to_remove, reverse=True):
            del self.objects[i]
            
        return step_reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw floor
        pygame.draw.line(self.screen, self.COLOR_FLOOR, (0, self.FLOOR_Y), (self.WIDTH, self.FLOOR_Y), 3)

        # Draw permanent lines
        for p1, p2 in self.lines:
            pygame.draw.aaline(self.screen, self.COLOR_LINE, p1, p2, 2)
            
        # Draw line in progress
        if self.is_drawing and self.current_line_start:
            pygame.draw.aaline(self.screen, self.COLOR_LINE_DRAWING, self.current_line_start, self.cursor_pos, 2)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw objects
        for obj in self.objects:
            obj.draw(self.screen)

        # Draw cursor
        cs = 8 # cursor size
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x - cs, self.cursor_pos.y), (self.cursor_pos.x + cs, self.cursor_pos.y), 2)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (self.cursor_pos.x, self.cursor_pos.y - cs), (self.cursor_pos.x, self.cursor_pos.y + cs), 2)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        saved_text = self.font_ui.render(f"Saved: {self.saved_objects_count} / {self.WIN_CONDITION}", True, self.COLOR_UI_TEXT)
        self.screen.blit(saved_text, (self.WIDTH - saved_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "saved_objects": self.saved_objects_count,
            "lines_drawn": len(self.lines)
        }
        
    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # To run, you will need to `pip install pygame`
    # It will open a window and you can play with the controls.
    
    # Un-comment the line below to run with a visible window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Chain Reaction Deflector")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # --- Manual Control Mapping ---
    # Arrows: Move cursor
    # Space: Hold to indicate you want to start a line
    # Shift: Hold to indicate you want to end a line
    #
    # The agent logic requires a state change, so you need to press and release.
    # To draw a line:
    # 1. Press and hold SPACE.
    # 2. Move the cursor with ARROWS.
    # 3. Press and hold SHIFT.
    # 4. Release both keys.
    # --------------------------------
    
    while not done:
        # Action defaults
        movement_action = 0 # none
        space_action = 0    # released
        shift_action = 0    # released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(60) # Run at 60 FPS for smooth manual play
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}")
    env.close()