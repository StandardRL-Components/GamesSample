import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:55:55.063224
# Source Brief: brief_01408.md
# Brief Index: 1408
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper Classes for Game Entities

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, pos, vel, radius, color, lifespan):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(vel)
        self.radius = radius
        self.color = color
        self.lifespan = lifespan
        self.initial_lifespan = lifespan

    def update(self):
        self.pos += self.vel
        self.lifespan -= 1
        self.radius = max(0, self.radius - 0.1)

    def draw(self, surface):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / self.initial_lifespan))
            color_with_alpha = (*self.color, alpha)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.radius), color_with_alpha)


class Ant:
    """Represents an ant with physical properties."""
    def __init__(self, x, y, size):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.size = size
        self.target_size = size
        self.mass = size
        self.max_speed = 6 - (size / 5) # Smaller ants are faster
        self.strength = size * 2 # Larger ants are stronger

    def update_properties(self):
        self.mass = self.size
        self.max_speed = max(1, 6 - (self.size / 5))
        self.strength = self.size * 2

    def update(self, friction, bounds_rect):
        # Smoothly interpolate size
        if abs(self.size - self.target_size) > 0.1:
            self.size += (self.target_size - self.size) * 0.2
            self.update_properties()

        # Simple physics
        self.vel *= friction
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)
        self.pos += self.vel

        # Boundary collision
        if self.pos.x - self.size < bounds_rect.left:
            self.pos.x = bounds_rect.left + self.size
            self.vel.x *= -0.5
        if self.pos.x + self.size > bounds_rect.right:
            self.pos.x = bounds_rect.right - self.size
            self.vel.x *= -0.5
        if self.pos.y - self.size < bounds_rect.top:
            self.pos.y = bounds_rect.top + self.size
            self.vel.y *= -0.5
        if self.pos.y + self.size > bounds_rect.bottom:
            self.pos.y = bounds_rect.bottom - self.size
            self.vel.y *= -0.5

    def draw(self, surface, is_selected):
        # Glow effect
        glow_radius = int(self.size * 1.5)
        glow_alpha = 60
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), glow_radius, (*GameEnv.COLOR_ANT, glow_alpha))
        
        # Main body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), int(self.size), GameEnv.COLOR_ANT)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.size), GameEnv.COLOR_ANT)
        
        # Highlight
        highlight_radius = int(self.size * 0.5)
        highlight_color = (180, 255, 180)
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), highlight_radius, highlight_color)

        if is_selected:
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.size) + 4, GameEnv.COLOR_UI_ACCENT)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), int(self.size) + 5, GameEnv.COLOR_UI_ACCENT)


class PhysicsObject:
    """Represents a movable block."""
    def __init__(self, x, y, width, height, mass):
        self.rect = pygame.Rect(x, y, width, height)
        self.pos = pygame.math.Vector2(self.rect.center)
        self.vel = pygame.math.Vector2(0, 0)
        self.mass = mass

    def apply_force(self, force):
        if self.mass > 0:
            self.vel += force / self.mass

    def update(self, friction, bounds_rect):
        self.vel *= friction
        self.pos += self.vel
        self.rect.center = self.pos

        # Boundary collision
        if self.rect.left < bounds_rect.left:
            self.rect.left = bounds_rect.left
            self.vel.x *= -0.5
        if self.rect.right > bounds_rect.right:
            self.rect.right = bounds_rect.right
            self.vel.x *= -0.5
        if self.rect.top < bounds_rect.top:
            self.rect.top = bounds_rect.top
            self.vel.y *= -0.5
        if self.rect.bottom > bounds_rect.bottom:
            self.rect.bottom = bounds_rect.bottom
            self.vel.y *= -0.5
        
        self.pos.update(self.rect.center)

    def draw(self, surface):
        pygame.draw.rect(surface, GameEnv.COLOR_OBJECT, self.rect, border_radius=3)
        pygame.draw.rect(surface, (120, 180, 255), self.rect.inflate(-4, -4), border_radius=3)

# Main Gymnasium Environment

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Command a colony of ants to solve physics-based puzzles. Grow and shrink ants to adjust their strength, then push blocks into targets before time runs out."
    )
    user_guide = (
        "Use arrow keys to select an ant. Hold space to grow an ant (stronger) or shift to shrink it (faster). Press space and shift together to start the simulation."
    )
    auto_advance = True

    # --- Colors and Constants ---
    WIDTH, HEIGHT = 640, 400
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (25, 30, 45)
    COLOR_ANT = (110, 220, 110)
    COLOR_OBJECT = (80, 140, 240)
    COLOR_TARGET = (220, 50, 50)
    COLOR_UI_TEXT = (230, 230, 230)
    COLOR_UI_ACCENT = (255, 255, 100)
    
    MIN_ANT_SIZE, MAX_ANT_SIZE = 5, 25
    FPS = 30
    MAX_STEPS = 2000
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 50, bold=True)

        self.bounds_rect = pygame.Rect(10, 10, self.WIDTH - 20, self.HEIGHT - 20)

        # Initialize state variables
        self.ants = []
        self.objects = []
        self.targets = []
        self.particles = []
        self.level = 0
        self.score = 0
        self.steps = 0
        self.timer = 0.0
        self.paused = True
        self.selected_ant_idx = 0
        self.prev_action = np.array([0, 0, 0])
        self.pause_cooldown = 0
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for debugging, not production

    def _setup_level(self):
        self.ants.clear()
        self.objects.clear()
        self.targets.clear()
        self.particles.clear()
        
        self.level += 1
        self.timer = max(30, 60 - (self.level // 2) * 5)
        num_objects = 1 + (self.level - 1) // 3
        
        # Place targets
        for i in range(num_objects):
            target_size = self.np_random.integers(30, 50)
            target_x = self.WIDTH - 80
            target_y = (self.HEIGHT / (num_objects + 1)) * (i + 1) - target_size / 2
            self.targets.append(pygame.Rect(target_x, target_y, target_size, target_size))

        # Place objects
        for i in range(num_objects):
            obj_size = self.targets[i].width
            obj_x = self.np_random.integers(50, 150)
            obj_y = self.np_random.integers(50, self.HEIGHT - 50)
            mass = (obj_size / 40) ** 2 * 10
            self.objects.append(PhysicsObject(obj_x, obj_y, obj_size, obj_size, mass))

        # Place ants
        num_ants = min(4, 2 + self.level // 2)
        for i in range(num_ants):
            ant_x = self.np_random.integers(self.WIDTH // 2 - 50, self.WIDTH // 2 + 50)
            ant_y = self.np_random.integers(50, self.HEIGHT - 50)
            self.ants.append(Ant(ant_x, ant_y, 15))
        
        self.selected_ant_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and 'level' in options:
            self.level = options['level'] - 1
        else:
            self.level = 0

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.paused = True
        self.pause_cooldown = 0
        self.prev_action = np.array([0, 0, 0])
        
        self._setup_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        terminated = False
        self.steps += 1
        
        # --- Handle Pause Toggle ---
        if self.pause_cooldown > 0: self.pause_cooldown -= 1
        
        # Pause is toggled by pressing space and shift simultaneously
        is_pause_press = space_held and shift_held
        was_pause_press = self.prev_action[1] == 1 and self.prev_action[2] == 1
        if is_pause_press and not was_pause_press and self.pause_cooldown == 0:
            self.paused = not self.paused
            self.pause_cooldown = 10 # 1/3 second cooldown

        # --- Handle Input ---
        if self.paused:
            # Ant Selection (on press)
            is_move_press = movement != 0 and self.prev_action[0] == 0
            if is_move_press and self.ants:
                if movement == 1: # Up
                    self.selected_ant_idx = (self.selected_ant_idx + 1) % len(self.ants)
                elif movement == 2: # Down
                    self.selected_ant_idx = (self.selected_ant_idx - 1 + len(self.ants)) % len(self.ants)
                elif movement == 4: # Right
                    self.selected_ant_idx = (self.selected_ant_idx + 1) % len(self.ants)
                elif movement == 3: # Left
                    self.selected_ant_idx = (self.selected_ant_idx - 1 + len(self.ants)) % len(self.ants)

            # Ant Size Change (on hold)
            if self.ants:
                selected_ant = self.ants[self.selected_ant_idx]
                size_changed = False
                if space_held and not shift_held:
                    selected_ant.target_size = min(self.MAX_ANT_SIZE, selected_ant.target_size + 0.5)
                    size_changed = True
                elif shift_held and not space_held:
                    selected_ant.target_size = max(self.MIN_ANT_SIZE, selected_ant.target_size - 0.5)
                    size_changed = True
                
                if size_changed and self.np_random.random() < 0.3:
                    # Sound effect placeholder: # sfx_size_change()
                    angle = self.np_random.uniform(0, 2 * math.pi)
                    speed = self.np_random.uniform(1, 3)
                    vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                    p_color = (200, 255, 200) if space_held else (255, 200, 200)
                    self.particles.append(Particle(selected_ant.pos, vel, 4, p_color, 20))

        # --- Update Game State (if not paused) ---
        if not self.paused:
            self.timer -= 1 / self.FPS
            reward -= 0.001 # Small time penalty per step

            # Store pre-physics distances for reward calculation
            pre_distances = [pygame.math.Vector2(obj.rect.center).distance_to(tgt.center) for obj, tgt in zip(self.objects, self.targets)]

            # Update ants and objects
            friction = 0.95
            for ant in self.ants:
                ant.update(friction, self.bounds_rect)
            for obj in self.objects:
                obj.update(friction, self.bounds_rect)

            # Physics Interactions
            self._handle_collisions()

            # Calculate distance-based reward
            post_distances = [pygame.math.Vector2(obj.rect.center).distance_to(tgt.center) for obj, tgt in zip(self.objects, self.targets)]
            for pre_d, post_d in zip(pre_distances, post_distances):
                # Reward for getting closer to target
                reward += (pre_d - post_d) * 0.1

        # Update particles regardless of pause state
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
        # --- Check for Termination Conditions ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        if terminated and not self.game_over:
            self.game_over = True
            # Sound effect placeholder: # sfx_win() or sfx_lose()
            # Spawn particles on win/loss
            for _ in range(100):
                pos = (self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT))
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(2, 6)
                vel = (math.cos(angle) * speed, math.sin(angle) * speed)
                p_color = self.COLOR_UI_ACCENT if term_reward > 0 else self.COLOR_TARGET
                self.particles.append(Particle(pos, vel, 8, p_color, 60))

        self.score += reward
        self.prev_action = action
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_collisions(self):
        # Ant-Object Collisions
        for ant in self.ants:
            for obj in self.objects:
                # AABB check first for performance
                if not ant.pos.x - ant.size > obj.rect.right and \
                   not ant.pos.x + ant.size < obj.rect.left and \
                   not ant.pos.y - ant.size > obj.rect.bottom and \
                   not ant.pos.y + ant.size < obj.rect.top:
                    
                    # More precise circle-rect collision check
                    closest_point = pygame.math.Vector2(
                        max(obj.rect.left, min(ant.pos.x, obj.rect.right)),
                        max(obj.rect.top, min(ant.pos.y, obj.rect.bottom))
                    )
                    dist_vec = ant.pos - closest_point
                    if dist_vec.length_squared() < ant.size * ant.size:
                        # Collision occurred
                        # Sound effect placeholder: # sfx_bump_soft()
                        push_force = dist_vec.normalize() * ant.strength * 0.1
                        obj.apply_force(push_force)
                        ant.vel -= push_force / ant.mass * 0.5 # Recoil

        # Object-Object Collisions
        for i, obj1 in enumerate(self.objects):
            for obj2 in self.objects[i+1:]:
                if obj1.rect.colliderect(obj2.rect):
                    # Sound effect placeholder: # sfx_bump_hard()
                    delta = pygame.math.Vector2(obj1.rect.center) - pygame.math.Vector2(obj2.rect.center)
                    dist = delta.length()
                    if dist == 0: continue
                    overlap = (obj1.rect.width / 2 + obj2.rect.width / 2) - dist
                    
                    # Resolve overlap
                    separation_vec = delta.normalize() * overlap
                    obj1.pos += separation_vec * 0.5
                    obj2.pos -= separation_vec * 0.5
                    
                    # Elastic collision response
                    v_rel = obj1.vel - obj2.vel
                    j = -(1.5) * v_rel.dot(delta.normalize()) / (1/obj1.mass + 1/obj2.mass)
                    impulse = j * delta.normalize()
                    
                    obj1.vel += impulse / obj1.mass
                    obj2.vel -= impulse / obj2.mass

    def _check_termination(self):
        # Win condition: all objects in their targets
        all_in_place = True
        if not self.objects: all_in_place = False

        for obj, target in zip(self.objects, self.targets):
            if not target.contains(obj.rect):
                all_in_place = False
                break
        
        if all_in_place:
            return True, 100.0

        # Lose condition: time runs out
        if self.timer <= 0:
            return True, -50.0

        # Lose condition: max steps reached
        if self.steps >= self.MAX_STEPS:
            return True, 0.0
            
        return False, 0.0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "timer": self.timer,
        }

    def _render_background(self):
        for x in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _render_game(self):
        # Draw targets
        for target in self.targets:
            surface = pygame.Surface(target.size, pygame.SRCALPHA)
            surface.fill((*self.COLOR_TARGET, 50))
            pygame.draw.rect(surface, (*self.COLOR_TARGET, 150), surface.get_rect(), 2, border_radius=4)
            self.screen.blit(surface, target.topleft)

        # Draw objects
        for obj in self.objects:
            obj.draw(self.screen)

        # Draw ants
        for i, ant in enumerate(self.ants):
            ant.draw(self.screen, is_selected=(i == self.selected_ant_idx and self.paused))

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

    def _render_ui(self):
        # Level
        level_text = self.font_main.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (20, 15))

        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 20, 15))

        # Timer
        timer_color = self.COLOR_UI_TEXT if self.timer > 10 else self.COLOR_TARGET
        timer_text = self.font_main.render(f"TIME: {max(0, self.timer):.1f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH // 2 - timer_text.get_width() // 2, 15))

        # Pause indicator
        if self.paused:
            pause_text = self.font_large.render("PLANNING PHASE", True, self.COLOR_UI_ACCENT)
            text_rect = pause_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            
            bg_rect = text_rect.inflate(20, 10)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((*self.COLOR_BG, 200))
            self.screen.blit(bg_surf, bg_rect)
            self.screen.blit(pause_text, text_rect)
            
            help_text_str = "[ARROWS] SELECT ANT  |  [SPACE] GROW  |  [SHIFT] SHRINK  |  [SPACE+SHIFT] SIMULATE"
            help_text = self.font_main.render(help_text_str, True, self.COLOR_UI_TEXT)
            self.screen.blit(help_text, (self.WIDTH // 2 - help_text.get_width() // 2, self.HEIGHT - 30))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    paused_manual = False
    
    # Use a dictionary to track held keys for smooth manual control
    keys_held = {
        'up': False, 'down': False, 'left': False, 'right': False,
        'space': False, 'shift': False
    }

    while running:
        # --- Map Pygame events to Gymnasium action space ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: keys_held['up'] = True
                if event.key == pygame.K_DOWN: keys_held['down'] = True
                if event.key == pygame.K_LEFT: keys_held['left'] = True
                if event.key == pygame.K_RIGHT: keys_held['right'] = True
                if event.key == pygame.K_SPACE: keys_held['space'] = True
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held['shift'] = True
                if event.key == pygame.K_r: # Reset button
                    obs, info = env.reset()
                if event.key == pygame.K_p: # Manual pause for inspection
                    paused_manual = not paused_manual
                if event.key == pygame.K_ESCAPE:
                    running = False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP: keys_held['up'] = False
                if event.key == pygame.K_DOWN: keys_held['down'] = False
                if event.key == pygame.K_LEFT: keys_held['left'] = False
                if event.key == pygame.K_RIGHT: keys_held['right'] = False
                if event.key == pygame.K_SPACE: keys_held['space'] = False
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: keys_held['shift'] = False

        if paused_manual:
            # Render but don't step the environment
            pygame.display.flip()
            env.clock.tick(env.FPS)
            continue

        # Construct the action from held keys
        movement = 0
        if keys_held['up']: movement = 1
        elif keys_held['down']: movement = 2
        elif keys_held['left']: movement = 3
        elif keys_held['right']: movement = 4
        
        space_btn = 1 if keys_held['space'] else 0
        shift_btn = 1 if keys_held['shift'] else 0

        action = np.array([movement, space_btn, shift_btn])

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        
        # Create a display if one doesn't exist
        try:
            display_surf = pygame.display.get_surface()
            if display_surf is None:
                raise Exception
            display_surf.blit(surf, (0, 0))
        except Exception:
            display_surf = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            display_surf.blit(surf, (0, 0))

        pygame.display.flip()
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset()

    env.close()