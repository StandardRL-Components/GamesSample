import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:33:52.664702
# Source Brief: brief_01172.md
# Brief Index: 1172
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent builds a coral reef.

    The agent controls a cursor on a grid and can place different types of
    coral. Placing corals with opposite magnetic polarity next to each other
    triggers a chain reaction of growth, increasing the reef's total area,
    which is the score. The goal is to maximize the reef area within a
    fixed number of steps.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Build a vibrant coral reef by strategically placing corals. Trigger chain reactions by matching "
        "polarities to maximize your reef's area and score."
    )
    user_guide = (
        "Use the arrow keys (↑↓←→) to move the cursor. Press 'space' to place a coral and 'shift' to cycle "
        "through available coral types."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    CELL_SIZE = 40
    MAX_STEPS = 1000
    UI_HEIGHT = 40

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_GRID = (20, 30, 50)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_UI_BG = (5, 10, 20)
    COLOR_UI_TEXT = (220, 220, 240)
    
    # Polarity
    P_POSITIVE = 1
    P_NEGATIVE = -1
    P_NEUTRAL = 0

    class Coral:
        """Represents a single coral on the board."""
        def __init__(self, grid_x, grid_y, species_info):
            self.id = random.randint(1, 1_000_000_000)
            self.grid_x = grid_x
            self.grid_y = grid_y
            self.species_info = species_info
            self.polarity = species_info['polarity']
            self.color = species_info['color']
            self.size = species_info['base_size']
            self.target_size = self.size
            self.animation_speed = 0.1

        def update(self):
            """Smoothly interpolates size for animation."""
            if self.size != self.target_size:
                self.size += (self.target_size - self.size) * self.animation_speed
        
        def get_pixel_pos(self):
            """Calculates the center pixel position."""
            return (
                int(self.grid_x * GameEnv.CELL_SIZE + GameEnv.CELL_SIZE / 2),
                int(self.grid_y * GameEnv.CELL_SIZE + GameEnv.CELL_SIZE / 2 + GameEnv.UI_HEIGHT)
            )

    class Particle:
        """Represents a visual effect particle."""
        def __init__(self, x, y, color):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            self.pos = [x, y]
            self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.lifespan = random.randint(15, 30)
            self.color = color
            self.radius = random.uniform(2, 4)

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifespan -= 1
            return self.lifespan > 0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self._define_coral_species()
        
    def _define_coral_species(self):
        """Defines all available coral types and their properties."""
        self.CORAL_SPECIES = [
            # Tier 0
            {'name': 'Anemone', 'polarity': self.P_POSITIVE, 'color': (100, 255, 150), 'base_size': 6, 'growth': 1.0, 'unlock_step': 0},
            {'name': 'Tube Coral', 'polarity': self.P_NEGATIVE, 'color': (255, 100, 100), 'base_size': 6, 'growth': 1.0, 'unlock_step': 0},
            # Tier 1
            {'name': 'Brain Coral', 'polarity': self.P_NEUTRAL, 'color': (200, 150, 255), 'base_size': 8, 'growth': 0.5, 'unlock_step': 200},
            # Tier 2
            {'name': 'Staghorn', 'polarity': self.P_POSITIVE, 'color': (50, 200, 255), 'base_size': 7, 'growth': 1.5, 'unlock_step': 400},
            {'name': 'Fire Coral', 'polarity': self.P_NEGATIVE, 'color': (255, 150, 50), 'base_size': 7, 'growth': 1.5, 'unlock_step': 600},
            # Tier 3
            {'name': 'Sun Coral', 'polarity': self.P_NEUTRAL, 'color': (255, 220, 50), 'base_size': 9, 'growth': 1.0, 'unlock_step': 800},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.board = {}  # {(x, y): Coral_instance}
        self.particles = []
        self.active_links = []
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.unlocked_species = []
        self._update_unlocked_species()
        
        self.selected_species_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False

        self.bonus_500_awarded = False
        self.bonus_800_awarded = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_held, shift_held)
        
        if space_held and not self.last_space_held:
            # SFX: place_coral.wav
            placement_reward = self._place_coral()
            reward += placement_reward

        self._update_animations()
        self.score = self._calculate_score()
        
        self._check_for_unlocks()
        reward += self._check_score_bonuses()

        self.steps += 1
        terminated = self.steps >= self.MAX_STEPS or len(self.board) >= self.GRID_COLS * self.GRID_ROWS
        if terminated:
            self.game_over = True
            
        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        """Processes agent actions."""
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        if shift_held and not self.last_shift_held:
            # SFX: ui_switch.wav
            if self.unlocked_species:
                self.selected_species_idx = (self.selected_species_idx + 1) % len(self.unlocked_species)

    def _place_coral(self):
        """Places a coral at the cursor and triggers reactions."""
        pos = tuple(self.cursor_pos)
        if pos in self.board:
            return -0.01 # Small penalty for invalid placement
        
        if not self.unlocked_species:
            return 0 # Cannot place anything
            
        species = self.unlocked_species[self.selected_species_idx]
        new_coral = self.Coral(pos[0], pos[1], species)
        self.board[pos] = new_coral
        
        return self._resolve_chain_reaction(new_coral)

    def _resolve_chain_reaction(self, start_coral):
        """Calculates and applies growth/merge chain reactions."""
        reward = 0.0
        q = deque([start_coral])
        processed_in_turn = {start_coral.id}
        
        while q:
            current_coral = q.popleft()
            
            # Check 4 neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = current_coral.grid_x + dx, current_coral.grid_y + dy
                neighbor = self.board.get((nx, ny))

                if neighbor and neighbor.id not in processed_in_turn:
                    # Growth Reaction: Opposite or Neutral-Polar
                    is_opposite = current_coral.polarity * neighbor.polarity == -1
                    is_neutral_interaction = abs(current_coral.polarity) + abs(neighbor.polarity) == 1
                    
                    if is_opposite or is_neutral_interaction:
                        # SFX: growth_sparkle.wav
                        current_coral.target_size += current_coral.species_info['growth']
                        neighbor.target_size += neighbor.species_info['growth']
                        reward += 0.1
                        
                        pos1 = current_coral.get_pixel_pos()
                        pos2 = neighbor.get_pixel_pos()
                        self.active_links.append((pos1, pos2, (255, 255, 255, 150)))
                        self._create_particles(pos1, neighbor.color, 5)
                        self._create_particles(pos2, current_coral.color, 5)
                        
                        processed_in_turn.add(neighbor.id)
                        q.append(neighbor)

                    # Merge Reaction: Matching non-neutral polarities
                    elif current_coral.polarity == neighbor.polarity and current_coral.polarity != 0:
                        # SFX: merge_blob.wav
                        current_area = math.pi * current_coral.size**2
                        neighbor_area = math.pi * neighbor.size**2
                        
                        # Current coral absorbs neighbor
                        current_coral.target_size = math.sqrt((current_area + neighbor_area) / math.pi)
                        
                        merge_pos = neighbor.get_pixel_pos()
                        del self.board[(nx, ny)]
                        
                        self._create_particles(merge_pos, neighbor.color, 20)
                        reward += 1.0
                        
                        # Do not add to processed_in_turn as it's removed
                        # Re-queue the merged coral to check for new interactions
                        if current_coral not in q:
                            q.append(current_coral)
        return reward

    def _update_animations(self):
        """Updates all animated elements."""
        self.active_links.clear() # Links only last one frame
        
        for coral in list(self.board.values()):
            coral.update()
            
        self.particles = [p for p in self.particles if p.update()]

    def _check_for_unlocks(self):
        """Unlocks new coral species based on step count."""
        if self._update_unlocked_species():
            # SFX: unlock.wav
            # Ensure selected index is valid after list changes
            if self.unlocked_species:
                self.selected_species_idx = min(self.selected_species_idx, len(self.unlocked_species) - 1)

    def _update_unlocked_species(self):
        """Helper to update the list of available species."""
        initial_len = len(self.unlocked_species)
        self.unlocked_species = [s for s in self.CORAL_SPECIES if self.steps >= s['unlock_step']]
        return len(self.unlocked_species) > initial_len

    def _check_score_bonuses(self):
        """Awards bonus rewards for reaching score milestones."""
        reward = 0
        if self.score > 500 and not self.bonus_500_awarded:
            reward += 50
            self.bonus_500_awarded = True
            # SFX: bonus_achieved.wav
        if self.score > 800 and not self.bonus_800_awarded:
            reward += 100
            self.bonus_800_awarded = True
            # SFX: bonus_achieved_major.wav
        return reward

    def _calculate_score(self):
        """Calculates total reef area."""
        total_area = sum(math.pi * c.size**2 for c in self.board.values())
        return int(total_area / 10) # Scale down for readability

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _render_game(self):
        """Renders the main game area (grid, corals, effects)."""
        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = r * self.CELL_SIZE + self.UI_HEIGHT
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.UI_HEIGHT), (x, self.SCREEN_HEIGHT))

        # Draw reaction links
        for start, end, color in self.active_links:
            pygame.draw.aaline(self.screen, color, start, end, True)

        # Draw corals
        for coral in list(self.board.values()):
            pos = coral.get_pixel_pos()
            radius = int(coral.size)
            if radius <= 0: continue
            
            # Glow effect
            glow_radius = int(radius * 1.5)
            glow_color = (*coral.color, 60) # RGBA
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (pos[0] - glow_radius, pos[1] - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Main body
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, coral.color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, coral.color)

        # Draw particles
        for p in self.particles:
            pygame.draw.circle(self.screen, p.color, p.pos, p.radius)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.cursor_pos[0] * self.CELL_SIZE,
            self.cursor_pos[1] * self.CELL_SIZE + self.UI_HEIGHT,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        """Renders the top UI bar."""
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (0, 0, self.SCREEN_WIDTH, self.UI_HEIGHT))
        
        # Score
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 8))
        
        # Steps
        steps_text = self.font_main.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(steps_text, (self.SCREEN_WIDTH - steps_text.get_width() - 10, 8))
        
        # Selected Card
        if self.unlocked_species:
            species = self.unlocked_species[self.selected_species_idx]
            
            # Preview box
            preview_x = self.SCREEN_WIDTH // 2 - 100
            pygame.draw.rect(self.screen, self.COLOR_GRID, (preview_x, 5, 200, 30), border_radius=5)
            
            # Polarity symbol
            polarity_map = {self.P_POSITIVE: "+", self.P_NEGATIVE: "-", self.P_NEUTRAL: "o"}
            symbol = polarity_map[species['polarity']]
            name_text = self.font_main.render(f"[{symbol}] {species['name']}", True, species['color'])
            self.screen.blit(name_text, (preview_x + 100 - name_text.get_width()//2, 8))

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            self.particles.append(self.Particle(pos[0], pos[1], color))

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage & Manual Play ---
    # Set a real video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Coral Reef Builder")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Track held keys to implement one action per press
    last_shift_pressed = False
    last_space_pressed = False
    
    while running:
        # Action defaults
        movement = 0 # none
        space = 0    # released
        shift = 0    # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                total_reward = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        # Since auto_advance is False, we should only register one action per key press
        # The environment handles this internally with last_space_held etc.
        # This logic here is for the manual play loop.
        current_space_pressed = keys[pygame.K_SPACE]
        if current_space_pressed:
            space = 1
            
        current_shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if current_shift_pressed:
            shift = 1
            
        action = np.array([movement, space, shift])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            print("Press 'R' to reset.")
            # Wait for reset command
            while True:
                reset_event = pygame.event.wait()
                if reset_event.type == pygame.QUIT:
                    running = False
                    break
                if reset_event.type == pygame.KEYDOWN and reset_event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    break
            
        clock.tick(30) # Limit frame rate for playability
        
    env.close()