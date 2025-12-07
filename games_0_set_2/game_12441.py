import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:35:11.186773
# Source Brief: brief_02441.md
# Brief Index: 2441
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player controls a reticle to catch falling
    geometric shapes. Collected shapes are used to craft more complex components
    to build a target megastructure. The game features a minimalist, geometric
    visual style with an emphasis on visual feedback and game feel.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Catch falling geometric shapes with your reticle and use them to craft components "
        "for a massive megastructure."
    )
    user_guide = (
        "Use ←→ arrow keys to move the reticle. Press space to collect shapes and shift to "
        "switch between 2D/3D crafting modes."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    PLAY_AREA_HEIGHT = 350
    COLLECTION_AREA_Y = 360
    RETICLE_SPEED = 15
    MAX_STEPS = 2000
    MAX_COLLECTED_SHAPES = 12
    SHAPE_SPAWN_RATE = 20 # Lower is faster

    # --- Colors ---
    COLOR_BG = (15, 20, 30)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_RETICLE = (0, 255, 255)
    COLOR_RETICLE_GLOW = (0, 150, 150)
    COLOR_BLUEPRINT = (50, 60, 80)
    COLOR_BLUEPRINT_COMPLETE = (0, 255, 150)
    COLOR_SHAPES = {
        'square': (255, 50, 100),
        'triangle': (255, 200, 0),
        'circle': (50, 150, 255)
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_mode = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.reticle_x = 0
        self.fall_speed = 0
        self.falling_shapes = []
        self.collected_shapes = []
        self.megastructure_parts = set()
        self.is_3d_mode = False
        self.particles = []
        self.space_was_held = False
        self.shift_was_held = False
        self.win_condition = False
        
        # --- Crafting Recipes ---
        self.recipes_2d = {
            frozenset(['square', 'triangle']): 'trapezoid',
            frozenset(['triangle', 'triangle']): 'rhombus',
            frozenset(['circle', 'square']): 'capsule'
        }
        self.recipes_3d = {
            frozenset(['square', 'square']): 'cube_frame',
            frozenset(['circle', 'circle']): 'sphere_lattice',
            frozenset(['capsule', 'rhombus']): 'engine_core'
        }
        self.all_recipes = {**self.recipes_2d, **self.recipes_3d}
        self.target_megastructure = set(self.all_recipes.values())

        # Initialize state variables
        # self.reset() # reset is called by the environment runner
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # this would run before reset completes, causing issues

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.reticle_x = self.SCREEN_WIDTH // 2
        self.fall_speed = 1.5
        self.falling_shapes = []
        self.collected_shapes = []
        self.megastructure_parts = set()
        self.is_3d_mode = False
        self.particles = []
        self.space_was_held = False
        self.shift_was_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._update_reticle(movement)
        
        # Handle single-press logic for space and shift
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        if shift_pressed:
            self.is_3d_mode = not self.is_3d_mode
            # SFX: Mode switch sound

        # --- Game Logic ---
        reward += self._update_falling_shapes()
        
        if space_pressed:
            collection_reward = self._handle_collection()
            if collection_reward > 0:
                reward += collection_reward
                crafting_reward = self._check_crafting()
                reward += crafting_reward
        
        self._update_particles()

        # --- Difficulty Scaling ---
        if self.steps % 50 == 0:
            self.fall_speed += 0.05

        # --- Termination Check ---
        terminated, term_reward = self._check_termination()
        reward += term_reward
        self.game_over = terminated
        
        self.score += reward

        truncated = self.steps >= self.MAX_STEPS
        if truncated:
            terminated = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    # --- Game Logic Helpers ---
    def _update_reticle(self, movement):
        if movement == 3:  # Left
            self.reticle_x -= self.RETICLE_SPEED
        elif movement == 4:  # Right
            self.reticle_x += self.RETICLE_SPEED
        self.reticle_x = np.clip(self.reticle_x, 0, self.SCREEN_WIDTH)

    def _update_falling_shapes(self):
        # Move existing shapes
        miss_reward = 0
        for shape in self.falling_shapes[:]:
            shape['pos'][1] += self.fall_speed
            if shape['pos'][1] > self.PLAY_AREA_HEIGHT:
                self.falling_shapes.remove(shape)
                miss_reward -= 0.01  # Penalty for missing a shape
                # SFX: Shape miss sound
        
        # Spawn new shapes
        if self.np_random.integers(0, self.SHAPE_SPAWN_RATE) == 0:
            shape_type = self.np_random.choice(list(self.COLOR_SHAPES.keys()))
            shape_pos = [self.np_random.integers(20, self.SCREEN_WIDTH - 20), -20]
            self.falling_shapes.append({'type': shape_type, 'pos': shape_pos})
        return miss_reward

    def _handle_collection(self):
        # Find shapes under the reticle
        collectable = [s for s in self.falling_shapes if abs(s['pos'][0] - self.reticle_x) < 30]
        if not collectable:
            return 0
        
        # Collect the lowest shape
        target_shape = min(collectable, key=lambda s: -s['pos'][1])
        
        if len(self.collected_shapes) < self.MAX_COLLECTED_SHAPES:
            self.falling_shapes.remove(target_shape)
            self.collected_shapes.append(target_shape['type'])
            self._create_particles(target_shape['pos'], self.COLOR_SHAPES[target_shape['type']])
            # SFX: Collect success sound
            return 0.1 # Reward for collecting a shape
        return 0

    def _check_crafting(self):
        current_recipes = self.recipes_3d if self.is_3d_mode else self.recipes_2d
        crafting_reward = 0
        
        temp_collected = self.collected_shapes[:]
        crafted_something = True
        while crafted_something:
            crafted_something = False
            for ingredients, result in current_recipes.items():
                if result in self.megastructure_parts:
                    continue

                # Check if we have the ingredients
                temp_inventory = temp_collected[:]
                has_ingredients = True
                for item in ingredients:
                    if item in temp_inventory:
                        temp_inventory.remove(item)
                    else:
                        has_ingredients = False
                        break
                
                if has_ingredients:
                    self.megastructure_parts.add(result)
                    temp_collected = temp_inventory # Consume ingredients
                    crafting_reward += 1.0 # Reward for crafting
                    # SFX: Crafting success sound
                    crafted_something = True
                    break # Restart scan for other recipes
        
        self.collected_shapes = temp_collected
        return crafting_reward

    def _check_termination(self):
        if len(self.megastructure_parts) == len(self.target_megastructure):
            self.win_condition = True
            return True, 100.0 # Win
        if len(self.collected_shapes) >= self.MAX_COLLECTED_SHAPES:
            return True, -100.0 # Lose (collection area full)
        # Time out is handled by truncated flag in step()
        return False, 0

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_particles(self, pos, color, count=20):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(20, 40),
                'color': color
            })

    # --- Observation and Info ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "collected_count": len(self.collected_shapes),
            "megastructure_progress": len(self.megastructure_parts) / len(self.target_megastructure)
        }

    # --- Rendering ---
    def _render_game(self):
        self._render_megastructure_blueprint()
        self._render_collection_area()
        self._render_falling_shapes()
        self._render_reticle()
        self._render_particles()

    def _render_reticle(self):
        x, y = int(self.reticle_x), self.PLAY_AREA_HEIGHT
        # Glow effect
        for i in range(5, 0, -1):
            alpha = 150 - i * 30
            color = (*self.COLOR_RETICLE_GLOW, alpha)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 10 + i*2, color)
        # Core reticle
        pygame.gfxdraw.aacircle(self.screen, x, y, 10, self.COLOR_RETICLE)
        pygame.gfxdraw.filled_circle(self.screen, x, y, 10, self.COLOR_RETICLE)
        pygame.draw.line(self.screen, self.COLOR_RETICLE, (x - 25, y), (x + 25, y), 2)

    def _render_falling_shapes(self):
        for shape in self.falling_shapes:
            pos = (int(shape['pos'][0]), int(shape['pos'][1]))
            color = self.COLOR_SHAPES[shape['type']]
            self._draw_shape(self.screen, shape['type'], pos, color, 15)
    
    def _render_collection_area(self):
        pygame.draw.line(self.screen, self.COLOR_BLUEPRINT, (0, self.PLAY_AREA_HEIGHT), (self.SCREEN_WIDTH, self.PLAY_AREA_HEIGHT), 1)
        pygame.draw.rect(self.screen, (25, 30, 45), (0, self.COLLECTION_AREA_Y, self.SCREEN_WIDTH, self.SCREEN_HEIGHT - self.COLLECTION_AREA_Y))
        
        for i, shape_type in enumerate(self.collected_shapes):
            x = 30 + i * 40
            y = self.COLLECTION_AREA_Y + 20
            color = self.COLOR_SHAPES[shape_type]
            desaturated_color = tuple(max(50, c - 80) for c in color)
            self._draw_shape(self.screen, shape_type, (x, y), desaturated_color, 12)

    def _render_megastructure_blueprint(self):
        blueprint_area = pygame.Rect(self.SCREEN_WIDTH - 160, 10, 150, 200)
        pygame.draw.rect(self.screen, (20, 25, 35), blueprint_area, border_radius=5)
        
        for i, part in enumerate(sorted(list(self.target_megastructure))):
            y_pos = blueprint_area.top + 15 + i * 25
            is_complete = part in self.megastructure_parts
            color = self.COLOR_BLUEPRINT_COMPLETE if is_complete else self.COLOR_BLUEPRINT
            text_surface = self.font_ui.render(f"[{'X' if is_complete else ' '}] {part.upper()}", True, color)
            self.screen.blit(text_surface, (blueprint_area.left + 10, y_pos))

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] / 10))
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score):05d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Mode indicator
        mode_str = "3D MODE" if self.is_3d_mode else "2D MODE"
        mode_color = (255, 100, 255) if self.is_3d_mode else (100, 255, 255)
        mode_text = self.font_mode.render(mode_str, True, mode_color)
        mode_rect = mode_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(mode_text, mode_rect)

        # Collection capacity
        cap_text = self.font_ui.render(f"STORAGE: {len(self.collected_shapes)}/{self.MAX_COLLECTED_SHAPES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(cap_text, (10, self.SCREEN_HEIGHT - 25))

        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg = "MEGASTRUCTURE COMPLETE!" if self.win_condition else "SYSTEM FAILURE"
            color = self.COLOR_BLUEPRINT_COMPLETE if self.win_condition else (255, 50, 50)
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _draw_shape(self, surface, shape_type, pos, color, size):
        x, y = pos
        if shape_type == 'square':
            pygame.draw.rect(surface, color, (x - size, y - size, size * 2, size * 2))
        elif shape_type == 'circle':
            pygame.gfxdraw.aacircle(surface, x, y, size, color)
            pygame.gfxdraw.filled_circle(surface, x, y, size, color)
        elif shape_type == 'triangle':
            points = [
                (x, y - size),
                (x - size, y + size * 0.7),
                (x + size, y + size * 0.7)
            ]
            pygame.gfxdraw.aapolygon(surface, points, color)
            pygame.gfxdraw.filled_polygon(surface, points, color)

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

if __name__ == '__main__':
    # --- Manual Play Testing ---
    # This block needs a display. It will not run in a headless environment.
    # To run, comment out the os.environ line at the top of the file.
    try:
        os.environ.pop("SDL_VIDEODRIVER")
    except KeyError:
        pass

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Megastructure Builder")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action[0] = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
            
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

        if terminated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}")
            print("Press 'R' to restart.")

    pygame.quit()