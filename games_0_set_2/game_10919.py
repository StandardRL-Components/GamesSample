import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:18:27.397400
# Source Brief: brief_00919.md
# Brief Index: 919
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import copy

class GameEnv(gym.Env):
    """
    Metabolic Chain Reaction Environment

    The player matches colored enzymes to substrates on a reaction pathway.
    Successful matches score points and advance the reaction.
    The goal is to achieve the highest score (metabolic efficiency) before
    running out of steps or time reversals.

    Action Space: MultiDiscrete([5, 2, 2])
    - action[0]: Movement (0=none, 1=up, 2=down, 3=left, 4=right) to select an enzyme.
    - action[1]: Match (0=released, 1=pressed) to use the selected enzyme.
    - action[2]: Time Reversal (0=released, 1=pressed) to undo the last move.

    Observation Space: Box(0, 255, (400, 640, 3), uint8)
    - An RGB image of the game screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Match colored enzymes to substrates on a reaction pathway to score points "
        "and unlock new, more complex enzymes."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select an enzyme, press space to match it, "
        "and use shift to trigger a time reversal."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 2000
    INITIAL_REVERSALS = 5
    PALETTE_SIZE = 6 # 2 rows of 3

    # --- Colors ---
    COLOR_BG = (10, 20, 30)
    COLOR_GRID = (20, 40, 60)
    COLOR_PATH = (40, 80, 120)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SELECTOR = (255, 255, 255)
    COLOR_COMPLETED = (80, 80, 90)
    
    ENZYME_COLORS = {
        "RED": (255, 80, 80),
        "GREEN": (80, 255, 80),
        "BLUE": (80, 120, 255),
        "YELLOW": (255, 255, 80),
        "PURPLE": (200, 80, 255),
        "CYAN": (80, 255, 255),
    }
    INITIAL_ENZYME_TYPES = ["RED", "GREEN", "BLUE"]
    UNLOCK_MILESTONES = {
        100: "YELLOW",
        500: "PURPLE",
        1000: "CYAN",
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_reversals_left = 0
        self.pathway = []
        self.active_substrate_index = 0
        self.enzyme_palette = []
        self.selector_pos = (0, 0) # (row, col)
        self.unlocked_enzymes = []
        self.particles = []
        self.history = deque(maxlen=20) # Store last 20 states for reversal

        # --- Input Handling ---
        self.last_space_held = False
        self.last_shift_held = False

        # --- Reward tracking ---
        self.reward_this_step = 0.0

        # --- Visual effects ---
        self.screen_shake = 0
        self.selector_render_pos = pygame.Vector2(0, 0)
        
        # self.reset() is called by the environment wrapper
        # self.validate_implementation() is for debugging and not needed in the final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_reversals_left = self.INITIAL_REVERSALS
        self.unlocked_enzymes = self.INITIAL_ENZYME_TYPES[:]
        
        self.active_substrate_index = 0
        self._generate_pathway()
        self._replenish_enzyme_palette()
        
        self.selector_pos = (0, 0)
        self.selector_render_pos = self._get_selector_target_pos()

        self.particles = []
        self.history.clear()
        self._save_state()

        self.last_space_held = False
        self.last_shift_held = False
        self.reward_this_step = 0.0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.reward_this_step = 0.0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

        if not self.game_over:
            self._handle_movement(movement)
            if space_press:
                self._handle_match()
            if shift_press:
                self._handle_reversal()
            
            self._update_game_state()
            
            old_score = self.score
            self.score += int(self.reward_this_step * 10) # Scale reward to score
            self.score = max(0, self.score)
            self._check_unlocks(old_score)

        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False, # Truncated is always False in this implementation
            self._get_info()
        )

    # --- Game Logic ---

    def _handle_movement(self, movement):
        row, col = self.selector_pos
        if movement == 1: # Up
            row = (row - 1) % 2
        elif movement == 2: # Down
            row = (row + 1) % 2
        elif movement == 3: # Left
            col = (col - 1) % (self.PALETTE_SIZE // 2)
        elif movement == 4: # Right
            col = (col + 1) % (self.PALETTE_SIZE // 2)
        self.selector_pos = (row, col)

    def _handle_match(self):
        selector_index = self.selector_pos[0] * (self.PALETTE_SIZE // 2) + self.selector_pos[1]
        selected_enzyme = self.enzyme_palette[selector_index]
        active_substrate = self.pathway[self.active_substrate_index]

        if selected_enzyme == active_substrate["type"]:
            # --- SUCCESSFUL MATCH ---
            self.reward_this_step += 1.0
            self._create_particles(active_substrate["pos"], selected_enzyme)
            
            active_substrate["state"] = "completed"
            self.active_substrate_index += 1

            if self.active_substrate_index >= len(self.pathway):
                self.reward_this_step += 2.0 # Bonus for completing chain
                self._generate_pathway()
                self.active_substrate_index = 0
            
            if self.active_substrate_index < len(self.pathway):
                self.pathway[self.active_substrate_index]["state"] = "active"
            self._replenish_enzyme_palette()
            self._save_state()
        else:
            # --- FAILED MATCH ---
            self.reward_this_step -= 0.1
            self.screen_shake = 10

    def _handle_reversal(self):
        if self.time_reversals_left > 0 and len(self.history) > 1:
            self.time_reversals_left -= 1
            self.history.pop() # Remove current state
            self._load_state(self.history[-1]) # Load previous state
            self.reward_this_step -= 0.5 # Small penalty for using reversal

    def _update_game_state(self):
        # Update particles
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

        # Update selector smooth movement
        target_pos = self._get_selector_target_pos()
        self.selector_render_pos = self.selector_render_pos.lerp(target_pos, 0.4)
        
        # Update screen shake
        if self.screen_shake > 0:
            self.screen_shake -= 1

    def _generate_pathway(self):
        self.pathway = []
        num_substrates = self.np_random.integers(5, 9)
        start_x = 100
        spacing = (self.SCREEN_WIDTH - 200) / (num_substrates -1) if num_substrates > 1 else 0
        y_center = 150
        y_amp = self.np_random.uniform(30, 60)
        freq = self.np_random.uniform(0.5, 1.5)
        phase = self.np_random.uniform(0, math.pi * 2)

        for i in range(num_substrates):
            x = start_x + i * spacing
            y = y_center + y_amp * math.sin(freq * i + phase)
            substrate_type = self.np_random.choice(self.unlocked_enzymes)
            self.pathway.append({
                "pos": pygame.Vector2(x, y),
                "type": substrate_type,
                "state": "inactive",
            })
        if self.pathway:
            self.pathway[0]["state"] = "active"

    def _replenish_enzyme_palette(self):
        self.enzyme_palette = []
        # Ensure the required enzyme is available
        if self.active_substrate_index < len(self.pathway):
            required_enzyme = self.pathway[self.active_substrate_index]["type"]
            self.enzyme_palette.append(required_enzyme)
        
        # Fill the rest of the palette
        while len(self.enzyme_palette) < self.PALETTE_SIZE:
            self.enzyme_palette.append(self.np_random.choice(self.unlocked_enzymes))
        
        self.np_random.shuffle(self.enzyme_palette)

    def _check_unlocks(self, old_score):
        for milestone, enzyme_type in self.UNLOCK_MILESTONES.items():
            if old_score < milestone <= self.score and enzyme_type not in self.unlocked_enzymes:
                self.unlocked_enzymes.append(enzyme_type)
                self.reward_this_step += 5.0 # Bonus for unlocking

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS or self.time_reversals_left < 0:
            self.game_over = True
        return self.game_over

    def _save_state(self):
        state = {
            "score": self.score,
            "active_substrate_index": self.active_substrate_index,
            "pathway": copy.deepcopy(self.pathway),
            "enzyme_palette": self.enzyme_palette[:],
            "unlocked_enzymes": self.unlocked_enzymes[:],
        }
        self.history.append(state)

    def _load_state(self, state):
        self.score = state["score"]
        self.active_substrate_index = state["active_substrate_index"]
        self.pathway = copy.deepcopy(state["pathway"])
        self.enzyme_palette = state["enzyme_palette"][:]
        self.unlocked_enzymes = state["unlocked_enzymes"][:]
        # Create a visual flash effect for reversal
        self.screen_shake = 5

    # --- Rendering ---

    def _get_observation(self):
        render_offset = pygame.Vector2(0, 0)
        if self.screen_shake > 0:
            render_offset.x = self.np_random.uniform(-self.screen_shake, self.screen_shake)
            render_offset.y = self.np_random.uniform(-self.screen_shake, self.screen_shake)

        self.screen.fill(self.COLOR_BG)
        self._render_background(render_offset)
        self._render_pathway(render_offset)
        self._render_enzymes(render_offset)
        self._render_selector(render_offset)
        self._render_particles(render_offset)
        self._render_ui() # UI is not affected by shake

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self, offset):
        for x in range(0, self.SCREEN_WIDTH, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x + offset.x, 0 + offset.y), (x + offset.x, self.SCREEN_HEIGHT + offset.y))
        for y in range(0, self.SCREEN_HEIGHT, 20):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0 + offset.x, y + offset.y), (self.SCREEN_WIDTH + offset.x, y + offset.y))

    def _render_pathway(self, offset):
        if len(self.pathway) > 1:
            points = [(s["pos"] + offset) for s in self.pathway]
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, points, 2)

        for i, substrate in enumerate(self.pathway):
            pos = substrate["pos"] + offset
            color = self.ENZYME_COLORS[substrate["type"]]
            
            if substrate["state"] == "completed":
                self._draw_glowing_circle(self.screen, self.COLOR_COMPLETED, pos, 10, 0)
            elif substrate["state"] == "active":
                glow = 15 + math.sin(pygame.time.get_ticks() * 0.005) * 5
                self._draw_glowing_circle(self.screen, color, pos, 12, glow)
            else: # inactive
                self._draw_glowing_circle(self.screen, color, pos, 10, 5)

    def _render_enzymes(self, offset):
        for i, enzyme_type in enumerate(self.enzyme_palette):
            row = i // (self.PALETTE_SIZE // 2)
            col = i % (self.PALETTE_SIZE // 2)
            x = self.SCREEN_WIDTH / 2 - 90 + col * 90
            y = self.SCREEN_HEIGHT - 60 + row * 40
            pos = pygame.Vector2(x, y) + offset
            color = self.ENZYME_COLORS[enzyme_type]
            self._draw_glowing_circle(self.screen, color, pos, 15, 10)

    def _render_selector(self, offset):
        pos = self.selector_render_pos + offset
        pygame.draw.rect(self.screen, self.COLOR_SELECTOR, (pos.x - 22, pos.y - 22, 44, 44), 2, border_radius=5)

    def _render_particles(self, offset):
        for p in self.particles:
            pos = p["pos"] + offset
            alpha = max(0, min(255, int(255 * (p["life"] / p["max_life"]))))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (pos.x - p["radius"], pos.y - p["radius"]))

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"Efficiency: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time Reversals
        reversal_text = self.font_small.render("Time Reversals:", True, self.COLOR_TEXT)
        self.screen.blit(reversal_text, (self.SCREEN_WIDTH - 200, 15))
        for i in range(self.INITIAL_REVERSALS):
            color = self.COLOR_TEXT if i < self.time_reversals_left else self.COLOR_GRID
            pygame.draw.circle(self.screen, color, (self.SCREEN_WIDTH - 70 + i * 20, 25), 6)
            pygame.draw.circle(self.screen, self.COLOR_BG, (self.SCREEN_WIDTH - 70 + i * 20, 25), 4)

    def _get_selector_target_pos(self):
        row, col = self.selector_pos
        x = self.SCREEN_WIDTH / 2 - 90 + col * 90
        y = self.SCREEN_HEIGHT - 60 + row * 40
        return pygame.Vector2(x, y)

    def _create_particles(self, pos, color_key):
        color = self.ENZYME_COLORS[color_key]
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(15, 30)
            self.particles.append({
                "pos": pos.copy(),
                "vel": vel,
                "life": life,
                "max_life": life,
                "color": color,
                "radius": self.np_random.uniform(2, 5)
            })
            
    def _draw_glowing_circle(self, surface, color, center, radius, glow_amount):
        # Draw glow layers
        if glow_amount > 0:
            for i in range(int(glow_amount // 2)):
                alpha = 80 * (1 - (i / (glow_amount / 2)))
                glow_radius = radius + i * 2
                pygame.gfxdraw.filled_circle(surface, int(center.x), int(center.y), int(glow_radius), (*color, int(alpha)))
                pygame.gfxdraw.aacircle(surface, int(center.x), int(center.y), int(glow_radius), (*color, int(alpha)))
        
        # Draw main circle
        pygame.gfxdraw.filled_circle(surface, int(center.x), int(center.y), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(center.x), int(center.y), int(radius), color)

    # --- Gymnasium Interface ---
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_reversals_left": self.time_reversals_left,
            "unlocked_enzymes": len(self.unlocked_enzymes)
        }

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # --- Manual Play Example ---
    # This part is for human play and debugging, not used by the gym environment
    # It requires a display, so we'll re-initialize pygame without the dummy driver
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.quit()
    pygame.init()
    pygame.font.init()
    
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a display for manual play
    pygame.display.set_caption("Metabolic Chain Reaction")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    action = [0, 0, 0] # [movement, space, shift]
    
    print("--- Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not terminated:
        # --- Get human input ---
        action = [0, 0, 0] # Reset actions
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
                # Map keys to movement actions
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # --- Render to display ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(GameEnv.FPS)

    env.close()
    print("Game Over!")