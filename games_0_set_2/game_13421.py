import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter

# --- Helper Classes for Visuals ---

class ElementParticle:
    """Represents a floating element inside the cell."""
    def __init__(self, letter, color, center, radius):
        self.letter = letter
        self.color = color
        self.cell_center = pygame.Vector2(center)
        self.cell_radius = radius - 20 # Keep particles from touching the edge
        
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, self.cell_radius)
        self.pos = self.cell_center + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist
        
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * random.uniform(0.5, 1.5)
        self.radius = 15

    def update(self):
        self.pos += self.vel
        # Bounce off the circular cell wall
        if self.pos.distance_to(self.cell_center) > self.cell_radius:
            self.pos -= self.vel # Step back
            normal = (self.pos - self.cell_center).normalize()
            self.vel = self.vel.reflect(normal)
            # Ensure it's inside after reflection
            while self.pos.distance_to(self.cell_center) > self.cell_radius:
                self.pos += self.vel * 0.1

    def draw(self, surface, font):
        # Glow effect
        for i in range(5, 0, -1):
            alpha = 80 - i * 15
            pygame.gfxdraw.filled_circle(
                surface, int(self.pos.x), int(self.pos.y),
                self.radius + i, (*self.color, alpha)
            )
        
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), self.radius, (255, 255, 255))
        
        text_surf = font.render(self.letter, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.pos)
        surface.blit(text_surf, text_rect)

class EffectParticle:
    """Represents a short-lived particle for reaction effects."""
    def __init__(self, pos, color):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(random.uniform(-3, 3), random.uniform(-3, 3))
        self.color = color
        self.max_lifespan = random.uniform(20, 40)
        self.lifespan = self.max_lifespan
        self.start_radius = random.uniform(3, 7)

    def update(self):
        self.lifespan -= 1
        self.pos += self.vel
        self.vel *= 0.95 # Damping

    def draw(self, surface):
        if self.lifespan > 0:
            progress = self.lifespan / self.max_lifespan
            radius = int(self.start_radius * progress)
            alpha = int(255 * progress)
            if radius > 0:
                pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), radius, (*self.color, alpha))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Combine basic elements according to hidden reaction rules to synthesize complex target molecules in a vibrant cellular environment."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to add elements to the formula. Press space to attempt a reaction and shift to clear the input."
    )
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Critical Gymnasium Interface ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width, self.screen_height = 640, 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # --- Visual & Game Constants ---
        self.FONT_UI = pygame.font.SysFont("Consolas", 18, bold=True)
        self.FONT_ELEMENT = pygame.font.SysFont("Arial", 18, bold=True)
        self.FONT_MSG = pygame.font.SysFont("Consolas", 24, bold=True)
        
        self.COLORS = {
            'BG': (10, 20, 35),
            'GRID': (20, 35, 55),
            'CELL_BORDER': (60, 120, 200),
            'UI_TEXT': (220, 220, 240),
            'SUCCESS': (60, 255, 120),
            'FAILURE': (255, 60, 60),
            'A': (230, 80, 80),   # Red
            'B': (80, 150, 230),  # Blue
            'C': (230, 80, 200),  # Magenta
            'D': (230, 160, 80),  # Orange
            'E': (80, 230, 150),  # Cyan
            'F': (220, 220, 80),  # Yellow
        }
        self.ELEMENT_MAP = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        self.CELL_CENTER = (self.screen_width // 2, self.screen_height // 2)
        self.CELL_RADIUS = 150
        self.MAX_STEPS = 1000
        self.REACTION_DURATION = 30 # steps

        # --- Game Rules ---
        self.REACTION_RULES = {
            frozenset(['A', 'B']): 'C',
            frozenset(['C', 'A']): 'D',
            frozenset(['B', 'B']): 'E',
            frozenset(['D', 'E']): 'F',
        }
        self.LEVELS = [
            {'target': 'C', 'inventory': {'A': 2, 'B': 2}},
            {'target': 'D', 'inventory': {'A': 3, 'B': 2}},
            {'target': 'E', 'inventory': {'A': 1, 'B': 3}},
            {'target': 'F', 'inventory': {'A': 3, 'B': 4}},
        ]
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_successes = 0
        self.current_level_idx = 0
        
        self.inventory = Counter()
        self.target_molecule = ''
        self.known_compounds = set()
        
        self.formula_input = []
        self.last_space_held = False
        self.last_shift_held = False

        self.element_particles = []
        self.effect_particles = []
        
        self.is_reacting = False
        self.reaction_timer = 0
        self.reaction_result = None
        self.reaction_reward = 0

        self.feedback_message = ""
        self.feedback_timer = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Difficulty progression
        self.current_level_idx = min(len(self.LEVELS) - 1, self.total_successes // 3)
        level_data = self.LEVELS[self.current_level_idx]
        
        self.inventory = Counter(level_data['inventory'])
        self.target_molecule = level_data['target']
        self.known_compounds = set(self.inventory.keys())
        
        self.formula_input = []
        self.last_space_held = False
        self.last_shift_held = False

        self.effect_particles = []
        self._create_element_visuals()

        self.is_reacting = False
        self.reaction_timer = 0
        self.reaction_result = None
        self.reaction_reward = 0
        
        self.feedback_message = "New Objective"
        self.feedback_timer = 60
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input ---
        if not self.is_reacting:
            # Add element to formula
            if movement in self.ELEMENT_MAP and len(self.formula_input) < 10:
                self.formula_input.append(self.ELEMENT_MAP[movement])
                # SFX: UI click
            
            # Clear formula (rising edge of shift)
            if shift_held and not self.last_shift_held:
                self.formula_input = []
                self.feedback_message = "Input Cleared"
                self.feedback_timer = 30
                # SFX: UI cancel
            
            # Trigger reaction (rising edge of space)
            if space_held and not self.last_space_held and self.formula_input:
                self.is_reacting = True
                self.reaction_timer = self.REACTION_DURATION
                self.reaction_reward, self.reaction_result = self._prepare_reaction()
                # SFX: Reaction start
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        # --- Update Game Logic ---
        self.steps += 1
        
        if self.is_reacting:
            self.reaction_timer -= 1
            if self.reaction_timer <= 0:
                reward_from_reaction = self._resolve_reaction()
                reward += reward_from_reaction
                self.is_reacting = False
        
        # Update animations
        for p in self.element_particles: p.update()
        for p in self.effect_particles: p.update()
        self.effect_particles = [p for p in self.effect_particles if p.lifespan > 0]
        if self.feedback_timer > 0: self.feedback_timer -= 1

        # --- Check Termination ---
        terminated = self._check_termination()
        if terminated:
            if self.inventory[self.target_molecule] > 0:
                reward += 100
                self.total_successes += 1
                self.feedback_message = "Target Synthesized!"
                self.feedback_timer = 120
            else: # Failure
                reward -= 100
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False, # Truncated
            self._get_info()
        )

    def _prepare_reaction(self):
        """Check formula and inventory, return potential reward and result."""
        formula_cost = Counter(self.formula_input)
        
        # Check if we have enough ingredients
        if any(self.inventory[elem] < count for elem, count in formula_cost.items()):
            self.feedback_message = "Insufficient Elements"
            self.feedback_timer = 60
            # SFX: Error buzz
            return 0, None # No reward, no result
        
        # Consume ingredients
        self.inventory -= formula_cost
        
        # Check reaction rules
        key = frozenset(self.formula_input)
        if key in self.REACTION_RULES:
            product = self.REACTION_RULES[key]
            return 0, product # Reward handled in resolve
        else:
            # Failed reaction, elements are wasted
            self.feedback_message = "Reaction Failed"
            self.feedback_timer = 60
            # SFX: Fizzle
            return -0.1 * len(self.formula_input), "FAIL"

    def _resolve_reaction(self):
        """Apply the result of the reaction to the game state. Returns reward."""
        product = self.reaction_result
        reward = self.reaction_reward
        
        if product and product != "FAIL":
            self.inventory[product] += 1
            # SFX: Synthesis success
            if product not in self.known_compounds:
                reward += 5
                self.known_compounds.add(product)
                self.feedback_message = f"New Compound: {product}"
                self.feedback_timer = 60
            else:
                self.feedback_message = f"Synthesized: {product}"
                self.feedback_timer = 60
            
            # Win condition check
            if product == self.target_molecule:
                self.game_over = True # This will be caught by _check_termination

            # Create success particles
            for _ in range(50):
                self.effect_particles.append(EffectParticle(self.CELL_CENTER, self.COLORS['SUCCESS']))

        elif product == "FAIL":
            # Create failure particles
            for _ in range(30):
                self.effect_particles.append(EffectParticle(self.CELL_CENTER, self.COLORS['FAILURE']))
        
        # Reset input and visuals
        self.formula_input = []
        self._create_element_visuals()
        
        return reward

    def _check_termination(self):
        if self.game_over: return True
        if self.steps >= self.MAX_STEPS: return True
        
        # Check for win condition (already checked in resolve)
        if self.inventory[self.target_molecule] > 0: return True
        
        # Check for loss condition (no possible valid reactions left)
        possible_reaction = False
        for ingredients in self.REACTION_RULES.keys():
            cost = Counter(ingredients)
            if all(self.inventory[elem] >= count for elem, count in cost.items()):
                possible_reaction = True
                break
        if not possible_reaction:
            return True
            
        return False

    def _create_element_visuals(self):
        self.element_particles.clear()
        for letter, count in self.inventory.items():
            if count > 0:
                for _ in range(count):
                    self.element_particles.append(
                        ElementParticle(letter, self.COLORS[letter], self.CELL_CENTER, self.CELL_RADIUS)
                    )

    def _get_observation(self):
        self.screen.fill(self.COLORS['BG'])
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "target": self.target_molecule,
            "inventory": dict(self.inventory),
            "level": self.current_level_idx,
            "total_successes": self.total_successes,
        }

    def _render_text(self, text, pos, font, color, center=False):
        # Render text with a dark outline for readability
        outline_color = (max(0, color[0]-100), max(0, color[1]-100), max(0, color[2]-100))
        text_surf = font.render(text, True, color)
        outline_surf = font.render(text, True, outline_color)
        
        if center:
            text_rect = text_surf.get_rect(center=pos)
        else:
            text_rect = text_surf.get_rect(topleft=pos)

        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            self.screen.blit(outline_surf, text_rect.move(dx, dy))
        self.screen.blit(text_surf, text_rect)

    def _render_game(self):
        # Draw background grid
        for i in range(0, self.screen_width, 20):
            pygame.draw.line(self.screen, self.COLORS['GRID'], (i, 0), (i, self.screen_height))
        for i in range(0, self.screen_height, 20):
            pygame.draw.line(self.screen, self.COLORS['GRID'], (0, i), (self.screen_width, i))

        # Draw cell with glow
        for i in range(15, 0, -2):
            alpha = 50 - i * 3
            pygame.gfxdraw.aacircle(self.screen, self.CELL_CENTER[0], self.CELL_CENTER[1], self.CELL_RADIUS + i, (*self.COLORS['CELL_BORDER'], alpha))
        pygame.gfxdraw.aacircle(self.screen, self.CELL_CENTER[0], self.CELL_CENTER[1], self.CELL_RADIUS, self.COLORS['CELL_BORDER'])
        
        # Draw particles
        for p in self.element_particles: p.draw(self.screen, self.FONT_ELEMENT)
        for p in self.effect_particles: p.draw(self.screen)

    def _render_ui(self):
        # --- Top Bar ---
        self._render_text(f"TARGET: {self.target_molecule}", (self.screen_width // 2, 20), self.FONT_UI, self.COLORS[self.target_molecule] if self.target_molecule in self.COLORS else self.COLORS['UI_TEXT'], center=True)
        self._render_text(f"SCORE: {self.score:.1f}", (10, 10), self.FONT_UI, self.COLORS['UI_TEXT'])
        self._render_text(f"STEP: {self.steps}/{self.MAX_STEPS}", (self.screen_width - 150, 10), self.FONT_UI, self.COLORS['UI_TEXT'])
        
        # --- Bottom Bar (Inventory) ---
        inv_text = "INVENTORY: "
        for elem in sorted(self.inventory.keys()):
            if self.inventory[elem] > 0:
                inv_text += f"{elem}: {self.inventory[elem]}  "
        self._render_text(inv_text, (10, self.screen_height - 25), self.FONT_UI, self.COLORS['UI_TEXT'])
        
        # --- Input Formula ---
        formula_str = "".join(self.formula_input)
        self._render_text(f"INPUT: {formula_str}", (self.screen_width // 2, self.screen_height - 55), self.FONT_UI, self.COLORS['UI_TEXT'], center=True)
        
        # --- Reaction Progress Bar ---
        if self.is_reacting:
            progress = 1 - (self.reaction_timer / self.REACTION_DURATION)
            bar_width = 200
            bar_height = 10
            bar_x = self.screen_width // 2 - bar_width // 2
            bar_y = self.screen_height // 2 + self.CELL_RADIUS + 15
            pygame.draw.rect(self.screen, self.COLORS['GRID'], (bar_x, bar_y, bar_width, bar_height), border_radius=3)
            pygame.draw.rect(self.screen, self.COLORS['SUCCESS'], (bar_x, bar_y, int(bar_width * progress), bar_height), border_radius=3)
            self._render_text("REACTING...", (self.screen_width // 2, bar_y + 20), self.FONT_UI, self.COLORS['UI_TEXT'], center=True)

        # --- Feedback Message ---
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 30.0)))
            color = (*self.COLORS['UI_TEXT'][:3], alpha)
            msg_surf = self.FONT_MSG.render(self.feedback_message, True, color)
            msg_rect = msg_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for display
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    display_screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Molecule Synthesis Environment")
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("Q: Quit")

    clock = pygame.time.Clock()
    while not done:
        # Map keyboard to MultiDiscrete action
        movement, space_held, shift_held = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                # Register movement as a single press event
                if event.key == pygame.K_UP: movement = 1
                elif event.key == pygame.K_DOWN: movement = 2
                elif event.key == pygame.K_LEFT: movement = 3
                elif event.key == pygame.K_RIGHT: movement = 4

        # Hold-down actions
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) 

    print("\n--- Game Over ---")
    print(f"Final Score: {env.score:.2f}")
    print(f"Final Info: {env._get_info()}")
    
    env.close()