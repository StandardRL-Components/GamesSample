import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:09:41.421416
# Source Brief: brief_00920.md
# Brief Index: 920
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a puzzle game about aligning chromosomes.
    The player controls a cursor to pick up, move, and drop color-coded
    chromosomes into their matching slots. Successful alignments can trigger
    chain reactions for bonus points, while mistakes lead to mutations.
    The goal is to complete the cell division by aligning all chromosomes.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Align color-coded chromosomes into their matching slots by controlling a cursor. "
        "Successful alignments can trigger chain reactions for bonus points."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move the cursor. Press space to pick up or drop a chromosome. "
        "Hold shift to move faster."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1500
    MUTATION_LIMIT = 5

    # Colors
    COLOR_BG = (10, 20, 40)
    COLOR_MEMBRANE = (30, 40, 70)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SUCCESS = (0, 255, 128)
    COLOR_ERROR = (255, 50, 50)
    CHROMOSOME_COLORS = [
        (50, 150, 255),  # Blue
        (255, 100, 100), # Red
        (80, 220, 120),  # Green
        (255, 180, 50),  # Orange
        (200, 100, 255), # Purple
        (50, 220, 220),  # Cyan
    ]

    # Game Parameters
    CURSOR_SPEED = 8
    CHROMOSOME_SIZE = 15
    SLOT_SIZE = 18
    PICKUP_RADIUS = 25
    SNAP_RADIUS = 30

    def __init__(self, render_mode="rgb_array", start_level=1):
        super().__init__()

        self.render_mode = render_mode
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # State variables
        self.start_level = start_level
        self.current_level = start_level
        self.cursor_pos = np.array([0.0, 0.0])
        self.chromosomes = []
        self.slots = []
        self.held_chromosome_idx = None
        self.space_was_held = False
        self.mutation_count = 0
        self.mutation_flash_timer = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options and options.get("new_game", False):
            self.current_level = self.start_level
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.mutation_count = 0
        self.mutation_flash_timer = 0
        self.held_chromosome_idx = None
        self.space_was_held = False
        self.cursor_pos = np.array([self.SCREEN_WIDTH / 4, self.SCREEN_HEIGHT / 2], dtype=float)
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Handle Input and Core Logic ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        action_reward = self._handle_input(movement, space_held, shift_held)
        reward += action_reward

        # --- Update Game State ---
        if self.mutation_flash_timer > 0:
            self.mutation_flash_timer -= 1
        
        # --- Check for Termination ---
        all_aligned = all(c['aligned'] for c in self.chromosomes)
        mutations_exceeded = self.mutation_count >= self.MUTATION_LIMIT
        timeout = self.steps >= self.MAX_STEPS

        terminated = all_aligned or mutations_exceeded or timeout
        truncated = False # Truncation is handled by the environment wrapper if needed
        if terminated:
            self.game_over = True
            if all_aligned:
                reward += 100.0  # Level complete bonus
                self.current_level += 1 # Progress to next level on next reset
                # SFX: Level Win
            if mutations_exceeded:
                reward -= 100.0  # Mutation limit penalty
                # SFX: Level Lose
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0.0
        
        # 1. Handle cursor movement
        speed = self.CURSOR_SPEED * 2 if shift_held else self.CURSOR_SPEED
        if movement == 1: self.cursor_pos[1] -= speed
        elif movement == 2: self.cursor_pos[1] += speed
        elif movement == 3: self.cursor_pos[0] -= speed
        elif movement == 4: self.cursor_pos[0] += speed
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.SCREEN_WIDTH)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.SCREEN_HEIGHT)
        
        # 2. Handle chromosome pickup/drop
        space_pressed = space_held and not self.space_was_held
        space_released = not space_held and self.space_was_held

        if space_pressed and self.held_chromosome_idx is None:
            # Try to pick up a chromosome
            for i, chromo in enumerate(self.chromosomes):
                if not chromo['aligned']:
                    dist = math.hypot(self.cursor_pos[0] - chromo['pos'][0], self.cursor_pos[1] - chromo['pos'][1])
                    if dist < self.PICKUP_RADIUS:
                        self.held_chromosome_idx = i
                        chromo['held'] = True
                        # SFX: Pickup
                        break
        
        elif space_released and self.held_chromosome_idx is not None:
            # Try to drop a chromosome
            chromo = self.chromosomes[self.held_chromosome_idx]
            chromo['held'] = False
            
            dropped_successfully = False
            for slot in self.slots:
                if slot['type'] == chromo['type'] and not slot['filled']:
                    dist = math.hypot(chromo['pos'][0] - slot['pos'][0], chromo['pos'][1] - slot['pos'][1])
                    if dist < self.SNAP_RADIUS:
                        # Successful alignment
                        chromo['pos'] = np.copy(slot['pos'])
                        chromo['aligned'] = True
                        slot['filled'] = True
                        reward += 1.0
                        # SFX: Success Snap
                        
                        # Check for chain reaction
                        partner_aligned = any(
                            c['aligned'] and c['type'] == chromo['type'] and c is not chromo 
                            for c in self.chromosomes
                        )
                        if partner_aligned:
                            reward += 5.0 # Chain reaction bonus
                            # SFX: Chain Reaction
                            
                        dropped_successfully = True
                        break
            
            if not dropped_successfully:
                # Failed alignment
                chromo['pos'] = np.copy(chromo['original_pos'])
                reward -= 1.0
                self.mutation_count += 1
                self.mutation_flash_timer = 5 # Flash for 5 frames
                # SFX: Failure
            
            self.held_chromosome_idx = None
            
        # 3. Update held chromosome position
        if self.held_chromosome_idx is not None:
            self.chromosomes[self.held_chromosome_idx]['pos'] = np.copy(self.cursor_pos)

        self.space_was_held = space_held
        return reward

    def _generate_level(self):
        self.chromosomes.clear()
        self.slots.clear()

        num_chromosome_pairs = 2 + (self.current_level - 1)
        num_types = min(num_chromosome_pairs, len(self.CHROMOSOME_COLORS))
        
        chromosome_zone = pygame.Rect(50, 50, self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - 100)
        slot_zone = pygame.Rect(self.SCREEN_WIDTH / 2 + 50, 50, self.SCREEN_WIDTH / 2 - 100, self.SCREEN_HEIGHT - 100)

        used_positions = []
        
        for i in range(num_types):
            color = self.CHROMOSOME_COLORS[i]
            for _ in range(2): # Create a pair
                # Create chromosome
                while True:
                    pos = np.array([
                        self.np_random.uniform(chromosome_zone.left, chromosome_zone.right),
                        self.np_random.uniform(chromosome_zone.top, chromosome_zone.bottom)
                    ])
                    if not any(math.hypot(pos[0]-p[0], pos[1]-p[1]) < self.CHROMOSOME_SIZE * 3 for p in used_positions):
                        used_positions.append(pos)
                        break
                self.chromosomes.append({
                    'pos': pos, 'original_pos': np.copy(pos), 'type': i, 'color': color,
                    'held': False, 'aligned': False
                })

                # Create slot
                while True:
                    pos = np.array([
                        self.np_random.uniform(slot_zone.left, slot_zone.right),
                        self.np_random.uniform(slot_zone.top, slot_zone.bottom)
                    ])
                    if not any(math.hypot(pos[0]-p[0], pos[1]-p[1]) < self.SLOT_SIZE * 3 for p in used_positions):
                        used_positions.append(pos)
                        break
                self.slots.append({'pos': pos, 'type': i, 'color': color, 'filled': False})

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_background()
        self._render_slots()
        self._render_chain_reactions()
        self._render_chromosomes()
        self._render_cursor()
        
        # Render UI overlay and effects
        self._render_effects()
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        pygame.gfxdraw.filled_circle(self.screen, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, 220, self.COLOR_MEMBRANE)
        pygame.gfxdraw.aacircle(self.screen, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, 220, self.COLOR_MEMBRANE)

    def _render_slots(self):
        for slot in self.slots:
            if not slot['filled']:
                self._draw_chromosome(self.screen, slot['pos'], slot['color'], self.SLOT_SIZE, outline=True)

    def _render_chain_reactions(self):
        aligned_by_type = {}
        for chromo in self.chromosomes:
            if chromo['aligned']:
                if chromo['type'] not in aligned_by_type:
                    aligned_by_type[chromo['type']] = []
                aligned_by_type[chromo['type']].append(chromo['pos'])
        
        for type_id, positions in aligned_by_type.items():
            if len(positions) == 2:
                pos1 = tuple(map(int, positions[0]))
                pos2 = tuple(map(int, positions[1]))
                color = self.CHROMOSOME_COLORS[type_id]
                self._draw_glowing_line(self.screen, color, pos1, pos2, 4)

    def _render_chromosomes(self):
        # Draw unheld chromosomes first
        for i, chromo in enumerate(self.chromosomes):
            if not chromo['held']:
                glow = chromo['aligned']
                self._draw_chromosome(self.screen, chromo['pos'], chromo['color'], self.CHROMOSOME_SIZE, glow=glow)
        # Draw held chromosome last so it's on top
        if self.held_chromosome_idx is not None:
            chromo = self.chromosomes[self.held_chromosome_idx]
            self._draw_chromosome(self.screen, chromo['pos'], chromo['color'], self.CHROMOSOME_SIZE, glow=True)

    def _render_cursor(self):
        pos = (int(self.cursor_pos[0]), int(self.cursor_pos[1]))
        size = 8
        # Glow effect
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size, (*self.COLOR_CURSOR, 100))
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], size, (*self.COLOR_CURSOR, 150))
        # Core cursor
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], size // 2, self.COLOR_CURSOR)

    def _render_effects(self):
        if self.mutation_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.mutation_flash_timer / 5))
            flash_surface.fill((*self.COLOR_ERROR, alpha))
            self.screen.blit(flash_surface, (0, 0))

        if self.action_space.sample()[2] == 1 and self.held_chromosome_idx is None: # A bit of a hack to check if shift could be held
             if any(action[2] == 1 for action in [self.action_space.sample() for _ in range(10)]): # Check if shift is held
                if self.steps % 2 == 0: # Animate the glow
                    glow_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surface, (*self.COLOR_CURSOR, 30), (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 10)
                    self.screen.blit(glow_surface, (0, 0))

    def _render_ui(self):
        # Mutations
        mutation_text = self.font_large.render(f"MUTATIONS: {self.mutation_count}/{self.MUTATION_LIMIT}", True, self.COLOR_TEXT)
        self.screen.blit(mutation_text, (20, 10))

        # Level and Score
        num_aligned = sum(1 for c in self.chromosomes if c['aligned'])
        total_chromos = len(self.chromosomes)
        progress_text = self.font_large.render(f"LEVEL {self.current_level} | ALIGNED: {num_aligned}/{total_chromos}", True, self.COLOR_TEXT)
        self.screen.blit(progress_text, (self.SCREEN_WIDTH - progress_text.get_width() - 20, 10))
        
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score:.0f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 35))

    def _draw_chromosome(self, surface, pos, color, size, outline=False, glow=False):
        x, y = int(pos[0]), int(pos[1])
        s = int(size)
        
        if glow:
            glow_color = (*color, 60)
            pygame.gfxdraw.filled_circle(surface, x, y, int(s * 1.8), glow_color)
            pygame.gfxdraw.aacircle(surface, x, y, int(s * 1.8), glow_color)

        if outline:
            width = 2
            pygame.draw.line(surface, color, (x - s, y - s), (x + s, y + s), width)
            pygame.draw.line(surface, color, (x - s, y + s), (x + s, y - s), width)
        else:
            width = max(5, int(s / 2.5))
            pygame.draw.line(surface, color, (x - s, y - s), (x + s, y + s), width)
            pygame.draw.line(surface, color, (x - s, y + s), (x + s, y - s), width)

    def _draw_glowing_line(self, surface, color, start, end, width):
        # Draw multiple transparent lines to create a glow effect
        for i in range(width, 0, -1):
            alpha = 150 - (i * 30)
            if alpha > 0:
                pygame.draw.line(surface, (*color, alpha), start, end, i * 2)
        pygame.draw.line(surface, (255, 255, 255), start, end, 1) # Core bright line

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.current_level,
            "mutations": self.mutation_count,
            "aligned": sum(1 for c in self.chromosomes if c['aligned'])
        }

    def close(self):
        pygame.quit()

# Example usage to test the environment visually
if __name__ == '__main__':
    # This block will not run in a strict headless environment
    # but is useful for local testing with a display.
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Mitosis Mayhem")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # No-op
        space_action = 0 # Released
        shift_action = 0 # Released
        
        # This event loop is for manual control, not for the agent
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement_action = 1
        elif keys[pygame.K_DOWN]: movement_action = 2
        elif keys[pygame.K_LEFT]: movement_action = 3
        elif keys[pygame.K_RIGHT]: movement_action = 4
        
        if keys[pygame.K_SPACE]: space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_action = 1
        
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Episode finished! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            obs, info = env.reset(options={"new_game": False}) # Continue to next level
            total_reward = 0
        
        clock.tick(30) # Run at 30 FPS
        
    env.close()