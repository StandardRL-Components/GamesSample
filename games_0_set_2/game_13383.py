import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:11:13.210976
# Source Brief: brief_03383.md
# Brief Index: 3383
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the agent must identify and clone repeating bolt
    sequences from a conveyor belt. The goal is to manage an inventory of cloned
    sequences, sell them for resources, and use those resources to upgrade an
    automation system, with the ultimate goal of reaching automation level 10.

    The gameplay involves pattern recognition, resource management, and strategic
    decision-making about when to clone and when to sell.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Identify and clone repeating bolt sequences from a conveyor belt. "
        "Sell cloned sequences for resources and upgrade your automation system to win."
    )
    user_guide = (
        "Controls: ↑/↓ to select an item in your inventory or the upgrade button. "
        "Press space to clone the sequence in the highlighted zone. "
        "Press shift to sell the selected item or purchase an upgrade."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 5000
    TARGET_AUTOMATION_LEVEL = 10

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (30, 45, 60)
    COLOR_CONVEYOR = (15, 20, 25)
    COLOR_UI_BG = (25, 40, 55)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (100, 110, 120)
    COLOR_SELECT_GLOW = (255, 255, 0)
    COLOR_UPGRADE_BUTTON = (0, 150, 200)
    COLOR_UPGRADE_BUTTON_HOVER = (50, 200, 255)

    BOLT_COLORS = [
        (50, 200, 255),  # Cyan
        (255, 100, 200), # Pink
        (255, 200, 50),  # Yellow-Orange
        (100, 255, 150), # Mint Green
        (200, 100, 255), # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.render_mode = render_mode

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_title = pygame.font.Font(None, 32)

        # --- Game State Initialization ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        # Game-specific state variables
        self.automation_level = 1
        self.resources = 0
        self.upgrade_cost = 0
        self.inventory = []
        self.inventory_capacity = 0
        self.total_sets_sold = 0

        self.sequence_length = 3
        self.current_bolt_types = []
        self.target_sequence = []

        self.conveyor_bolts = []
        self.conveyor_speed = 0.0
        self.next_spawn_x = 0.0

        self.selected_idx = 0 # 0 to N-1 for inventory, N for upgrade button

        self.last_space_held = False
        self.last_shift_held = False

        self.particles = []
        self.feedback_fx = []

        # self.reset() is called by the environment wrapper
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Game State ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False

        self.automation_level = 1
        self.resources = 10 # Start with some resources
        self.total_sets_sold = 0
        self.sequence_length = 3

        self._update_level_params()

        self.inventory = []
        self.conveyor_bolts = []
        self.next_spawn_x = self.SCREEN_WIDTH + 100
        self._generate_target_sequence()

        self.selected_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.particles = []
        self.feedback_fx = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- Unpack Action ---
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1

        # --- Handle Input (Edge-Triggered) ---
        is_space_trigger = space_pressed and not self.last_space_held
        is_shift_trigger = shift_pressed and not self.last_shift_held

        # Navigation
        # Simplified: Up/Down cycles through all selectable items (inventory + upgrade)
        if movement in [1, 2]: # Up or Down
            num_selectable = len(self.inventory) + 1
            if num_selectable > 0:
                direction = -1 if movement == 1 else 1
                self.selected_idx = (self.selected_idx + direction) % num_selectable

        # Action: Clone (Space)
        if is_space_trigger:
            reward += self._execute_clone()

        # Action: Sell/Upgrade (Shift)
        if is_shift_trigger:
            reward += self._execute_sell_or_upgrade()

        self.last_space_held = space_pressed
        self.last_shift_held = shift_pressed

        # --- Update Game Logic ---
        self._update_conveyor()
        self._update_particles()
        self._update_feedback_fx()

        # --- Check Termination ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated or truncated:
            self.game_over = True
            if self.automation_level >= self.TARGET_AUTOMATION_LEVEL:
                reward += 100.0  # Win bonus
            else:
                reward -= 50.0  # Loss penalty

        self.score += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_clone(self):
        # Sound: CLONE_ATTEMPT
        if len(self.inventory) >= self.inventory_capacity:
            # Sound: ERROR_BEEP
            self._add_feedback_fx('clone_fail', (150, 350), 'Inventory Full', (255, 50, 50))
            return -1.0 # Penalty for trying to clone with full inventory

        clone_zone_x = 150
        cloned_sequence = [b['type'] for b in self.conveyor_bolts if clone_zone_x - 15 < b['x'] < clone_zone_x + (self.sequence_length * 30) - 15]

        if len(cloned_sequence) != self.sequence_length:
            self._add_feedback_fx('clone_fail', (150, 350), 'Bad Timing', (255, 150, 0))
            return -2.0 # Penalty for mis-timed clone

        # Compare sequences and calculate reward
        correct_bolts = 0
        for i in range(self.sequence_length):
            if cloned_sequence[i] == self.target_sequence[i]:
                correct_bolts += 1

        is_perfect_match = (correct_bolts == self.sequence_length)
        reward = (correct_bolts * 1.0) - ((self.sequence_length - correct_bolts) * 0.5)

        self.inventory.append({'seq': cloned_sequence, 'is_correct': is_perfect_match})

        if is_perfect_match:
            # Sound: CLONE_SUCCESS
            reward += 5.0 # Bonus for a perfect set
            self._add_feedback_fx('clone_ok', (clone_zone_x, 320), 'PERFECT CLONE', (50, 255, 50))
            self._generate_target_sequence() # New sequence to find
        else:
            # Sound: CLONE_PARTIAL_FAIL
            self._add_feedback_fx('clone_fail', (clone_zone_x, 320), f'{correct_bolts}/{self.sequence_length} MATCH', (255, 255, 50))

        # Visual effect: particles from clone zone to inventory
        for i in range(10):
            self._add_particle((clone_zone_x + 15, 360), (random.uniform(-2, 2), random.uniform(-5, -3)), self.BOLT_COLORS[cloned_sequence[0] % len(self.BOLT_COLORS)], 30)

        return reward

    def _execute_sell_or_upgrade(self):
        # Check if Upgrade button is selected
        if self.selected_idx == len(self.inventory):
            if self.resources >= self.upgrade_cost:
                # Sound: UPGRADE_SUCCESS
                self.resources -= self.upgrade_cost
                self.automation_level += 1
                self._update_level_params()
                self._add_feedback_fx('upgrade', (self.SCREEN_WIDTH-100, self.SCREEN_HEIGHT-50), 'LEVEL UP!', (50, 255, 255))
                # Visual effect for upgrade
                for _ in range(50):
                    self._add_particle((self.SCREEN_WIDTH-100, self.SCREEN_HEIGHT-50), (random.uniform(-4, 4), random.uniform(-4, 4)), self.COLOR_SELECT_GLOW, 40)
                return 10.0 # Reward for upgrading
            else:
                # Sound: ERROR_BEEP
                self._add_feedback_fx('fail', (self.SCREEN_WIDTH-100, self.SCREEN_HEIGHT-50), 'NEED RESOURCES', (255, 50, 50))
                return -0.5 # Small penalty for trying to upgrade without funds

        # Otherwise, sell inventory item
        elif len(self.inventory) > 0 and self.selected_idx < len(self.inventory):
            item_to_sell = self.inventory.pop(self.selected_idx)

            # Clamp selected_idx to new valid range
            if self.selected_idx >= len(self.inventory) and len(self.inventory) > 0:
                self.selected_idx = len(self.inventory) - 1

            value = 0
            if item_to_sell['is_correct']:
                # Sound: SELL_SUCCESS
                value = self.sequence_length * 2
                self.total_sets_sold += 1
                if self.total_sets_sold % 5 == 0 and self.sequence_length < 10:
                    self.sequence_length += 1
            else:
                # Sound: SELL_JUNK
                value = 1 # Sell junk to clear space

            self.resources += value

            # Visual effect for selling
            inv_pos = (480, 50 + self.selected_idx * 25)
            res_pos = (self.SCREEN_WIDTH - 150, 50)
            for _ in range(15):
                vel = ((res_pos[0]-inv_pos[0])/30 + random.uniform(-1,1), (res_pos[1]-inv_pos[1])/30 + random.uniform(-1,1))
                self._add_particle(inv_pos, vel, (200, 200, 50), 30)

            return 0.5 # Small reward for any sale action

        return 0.0

    def _update_level_params(self):
        self.upgrade_cost = 20 + (self.automation_level ** 2) * 5
        self.inventory_capacity = 5 + self.automation_level
        self.conveyor_speed = 1.0 + self.automation_level * 0.2

        num_types = 2
        if self.automation_level >= 7:
            num_types = 4
        elif self.automation_level >= 3:
            num_types = 3
        self.current_bolt_types = list(range(num_types))

    def _generate_target_sequence(self):
        self.target_sequence = [self.np_random.choice(self.current_bolt_types) for _ in range(self.sequence_length)]

    def _update_conveyor(self):
        # Move existing bolts
        for bolt in self.conveyor_bolts:
            bolt['x'] -= self.conveyor_speed

        # Remove off-screen bolts
        self.conveyor_bolts = [b for b in self.conveyor_bolts if b['x'] > -20]

        # Spawn new bolts
        self.next_spawn_x -= self.conveyor_speed
        if self.next_spawn_x <= self.SCREEN_WIDTH:
            # Spawn a random sequence
            seq = [self.np_random.choice(self.current_bolt_types) for _ in range(self.np_random.integers(3, 7))]
            for i, bolt_type in enumerate(seq):
                self.conveyor_bolts.append({'x': self.next_spawn_x + i * 30, 'type': bolt_type})
            self.next_spawn_x += len(seq) * 30 + self.np_random.integers(90, 200)

    def _check_termination(self):
        if self.automation_level >= self.TARGET_AUTOMATION_LEVEL:
            return True # Win condition

        # Loss condition: inventory is full of incorrect sequences
        if len(self.inventory) >= self.inventory_capacity:
            if all(not item['is_correct'] for item in self.inventory):
                return True
        return False

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "automation_level": self.automation_level,
            "resources": self.resources,
            "inventory_size": len(self.inventory),
            "target_sequence_length": self.sequence_length,
        }

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background_grid()
        self._render_conveyor()
        self._render_ui()
        self._render_particles()
        self._render_feedback_fx()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._get_observation()
        return None

    def _render_background_grid(self):
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_conveyor(self):
        # Belt
        belt_y, belt_h = 340, 50
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, (0, belt_y, self.SCREEN_WIDTH, belt_h))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, belt_y), (self.SCREEN_WIDTH, belt_y), 2)
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, belt_y + belt_h), (self.SCREEN_WIDTH, belt_y + belt_h), 2)

        # Clone Zone
        zone_x, zone_w = 150, self.sequence_length * 30
        pygame.gfxdraw.rectangle(self.screen, (zone_x, belt_y, zone_w, belt_h), (*self.COLOR_SELECT_GLOW, 100))

        # Bolts
        for bolt in self.conveyor_bolts:
            self._draw_bolt(int(bolt['x']), belt_y + belt_h // 2, bolt['type'])

    def _draw_bolt(self, x, y, bolt_type, radius=12):
        color = self.BOLT_COLORS[bolt_type % len(self.BOLT_COLORS)]
        darker_color = tuple(c * 0.7 for c in color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius, darker_color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, radius - 2, color)
        pygame.gfxdraw.aacircle(self.screen, x, y, radius, darker_color)

    def _render_ui(self):
        # --- Main UI Panel (Right Side) ---
        ui_x, ui_y, ui_w, ui_h = 450, 10, 180, self.SCREEN_HEIGHT - 20
        pygame.gfxdraw.box(self.screen, (ui_x, ui_y, ui_w, ui_h), (*self.COLOR_UI_BG, 200))
        pygame.gfxdraw.rectangle(self.screen, (ui_x, ui_y, ui_w, ui_h), (*self.COLOR_GRID, 255))

        # Title
        self._draw_text("INVENTORY", (ui_x + ui_w / 2, ui_y + 20), self.font_main, self.COLOR_TEXT, center=True)

        # Inventory Slots
        for i, item in enumerate(self.inventory):
            slot_y = ui_y + 45 + i * 25
            is_selected = (i == self.selected_idx)
            if is_selected:
                pygame.gfxdraw.box(self.screen, (ui_x + 5, slot_y - 8, ui_w - 10, 20), (*self.COLOR_SELECT_GLOW, 50))
                pygame.gfxdraw.rectangle(self.screen, (ui_x + 5, slot_y - 8, ui_w - 10, 20), self.COLOR_SELECT_GLOW)

            for j, bolt_type in enumerate(item['seq']):
                self._draw_bolt(ui_x + 15 + j * 16, slot_y, bolt_type, radius=6)

            if not item['is_correct']:
                pygame.draw.line(self.screen, (255,50,50), (ui_x + 10, slot_y), (ui_x + 10 + len(item['seq']) * 16, slot_y), 2)

        # Inventory Capacity
        cap_text = f"{len(self.inventory)} / {self.inventory_capacity}"
        self._draw_text(cap_text, (ui_x + ui_w - 10, ui_y + 20), self.font_main, self.COLOR_TEXT_DIM, align='right')

        # Upgrade Button
        upgrade_y = ui_y + ui_h - 60
        is_selected = self.selected_idx == len(self.inventory)
        btn_color = self.COLOR_UPGRADE_BUTTON_HOVER if is_selected else self.COLOR_UPGRADE_BUTTON
        pygame.draw.rect(self.screen, btn_color, (ui_x + 10, upgrade_y, ui_w - 20, 50), border_radius=5)
        if is_selected:
            pygame.draw.rect(self.screen, self.COLOR_SELECT_GLOW, (ui_x + 10, upgrade_y, ui_w - 20, 50), 2, border_radius=5)

        self._draw_text(f"UPGRADE LVL {self.automation_level+1}", (ui_x + ui_w/2, upgrade_y + 15), self.font_main, self.COLOR_TEXT, center=True)
        self._draw_text(f"Cost: {self.upgrade_cost} Res", (ui_x + ui_w/2, upgrade_y + 35), self.font_small, self.COLOR_TEXT, center=True)

        # --- Top Info Bar ---
        info_w, info_h = 440, 60
        pygame.gfxdraw.box(self.screen, (5, 5, info_w, info_h), (*self.COLOR_UI_BG, 200))
        pygame.gfxdraw.rectangle(self.screen, (5, 5, info_w, info_h), (*self.COLOR_GRID, 255))

        # Target Sequence
        self._draw_text("TARGET SEQUENCE:", (20, 20), self.font_small, self.COLOR_TEXT_DIM)
        for i, bolt_type in enumerate(self.target_sequence):
            self._draw_bolt(30 + i * 35, 45, bolt_type, radius=12)

        # Stats
        self._draw_text(f"RESOURCES: {self.resources}", (250, 20), self.font_main, (220, 220, 100))
        self._draw_text(f"AUTO-LEVEL: {self.automation_level}", (250, 45), self.font_main, self.COLOR_TEXT)

    def _draw_text(self, text, pos, font, color, center=False, align='left'):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        elif align == 'right':
            text_rect.topright = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    # --- Visual Effects ---

    def _add_particle(self, pos, vel, color, life):
        self.particles.append({'pos': list(pos), 'vel': list(vel), 'life': life, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * 8)))
            color = (*p['color'], alpha)
            size = int(p['life'] / 10) + 1
            pygame.draw.circle(self.screen, color, (int(p['pos'][0]), int(p['pos'][1])), size)

    def _add_feedback_fx(self, fxtype, pos, text, color):
        self.feedback_fx.append({'type': fxtype, 'pos': pos, 'text': text, 'color': color, 'life': 60})

    def _update_feedback_fx(self):
        for fx in self.feedback_fx:
            fx['life'] -= 1
            fx['pos'] = (fx['pos'][0], fx['pos'][1] - 0.5)
        self.feedback_fx = [fx for fx in self.feedback_fx if fx['life'] > 0]

    def _render_feedback_fx(self):
        for fx in self.feedback_fx:
            alpha = max(0, min(255, int(fx['life'] * 4.25)))
            color = (*fx['color'], alpha)

            text_surface = self.font_main.render(fx['text'], True, color)
            text_rect = text_surface.get_rect(center=fx['pos'])
            self.screen.blit(text_surface, text_rect)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It is NOT used by the platform for evaluation, but can be helpful for debugging.
    
    # Un-comment the line below to run with a graphical display
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Bolt Cloner")
    clock = pygame.time.Clock()
    
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n--- Manual Controls ---")
    print(GameEnv.user_guide)
    print("Q: Quit")
    
    while not terminated and not truncated:
        # --- Action Mapping for Human Play ---
        movement = 0 # none
        space = 0 # released
        shift = 0 # released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render to Screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Print Info and Control FPS ---
        if env.steps % 30 == 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Level: {info['automation_level']}, Res: {info['resources']}")
        
        clock.tick(30) # Run at 30 FPS

    print(f"\nGame Over! Final Score: {total_reward:.2f}")
    env.close()