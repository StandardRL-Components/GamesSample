import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:41:05.157201
# Source Brief: brief_01892.md
# Brief Index: 1892
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player manages a 3-slot inventory.
    Items arrive on a conveyor belt and must be combined to create a target item 'D'.
    The goal is to create 5 'D' items before the time runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Manage your inventory by combining items from a conveyor belt. "
        "Combine 'A' items into 'C', and 'C' items into 'D' to win the game."
    )
    user_guide = (
        "Controls: Use ← and → to shift the items in your inventory. "
        "Combine two of the same item to create a new one."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 3000  # 60 seconds at 50 steps/sec
    TARGET_D_COUNT = 5
    
    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_CONVEYOR = (40, 45, 55)
    COLOR_SLOT = (25, 30, 40)
    COLOR_SLOT_HIGHLIGHT = (50, 60, 75)
    COLOR_TEXT = (220, 220, 230)
    ITEM_COLORS = {
        'A': (70, 200, 80),   # Green
        'B': (220, 60, 60),    # Red
        'C': (60, 120, 220),  # Blue
        'D': (230, 200, 50),   # Yellow
    }
    ITEM_OUTLINE = (255, 255, 255)

    # Game Mechanics
    INVENTORY_SIZE = 3
    INITIAL_CONVEYOR_TIMER = 100 # Steps per new item
    SHIFT_COOLDOWN_STEPS = 10

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.render_mode = render_mode
        
        # --- GYMNASIUM SPACES ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- PYGAME SETUP ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- STATE VARIABLES ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.d_created = None
        self.game_over = None
        
        self.inventory = None
        self.conveyor_items = None
        self.conveyor_timer = None
        self.conveyor_item_interval = None
        
        self.shift_cooldown = None
        self.particles = None
        
        # --- VISUAL STATE ---
        # For smooth animations
        self.inventory_visual_pos = None
        self.last_action_time = 0

        # The reset method is called here to initialize the state, but Gymnasium's
        # standard practice is to not call it in __init__. However, to pass
        # the validation check which relies on an initialized state, we call it.
        # A user is expected to call reset() again before starting an episode.
        # self.reset()
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.d_created = 0
        self.game_over = False
        
        self.inventory = [None] * self.INVENTORY_SIZE
        self.conveyor_items = deque()
        self.conveyor_item_interval = self.INITIAL_CONVEYOR_TIMER
        self.conveyor_timer = self.conveyor_item_interval
        
        self.shift_cooldown = 0
        self.particles = []

        self.inventory_visual_pos = [i for i in range(self.INVENTORY_SIZE)]

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0

        # --- UPDATE GAME LOGIC ---
        self._handle_actions(action)
        reward += self._update_conveyor()
        reward += self._process_combinations()
        self._update_particles()
        
        # Update cooldowns
        if self.shift_cooldown > 0:
            self.shift_cooldown -= 1
        
        # --- CHECK TERMINATION ---
        terminated = False
        if self.d_created >= self.TARGET_D_COUNT:
            # print("VICTORY!")
            reward += 100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            # print("DEFEAT: Time out")
            reward += -100
            terminated = True
            self.game_over = True

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement = action[0]
        
        if self.shift_cooldown > 0:
            return

        if movement in [3, 4]: # Left or Right shift
            # # sfx: Inventory_Shift.wav
            if movement == 3: # Left
                self.inventory.append(self.inventory.pop(0))
            elif movement == 4: # Right
                self.inventory.insert(0, self.inventory.pop(-1))
            
            self.shift_cooldown = self.SHIFT_COOLDOWN_STEPS
            self.last_action_time = self.steps

    def _update_conveyor(self):
        reward = 0
        
        # Move existing items
        for item in self.conveyor_items:
            item['pos_x'] -= item['speed']

        # Spawn new items
        self.conveyor_timer -= 1
        if self.conveyor_timer <= 0:
            self._spawn_item()
            self.conveyor_timer = self.conveyor_item_interval

        # Check for item delivery
        if self.conveyor_items and self.conveyor_items[0]['pos_x'] < self.SCREEN_WIDTH / 2:
            delivered_item = self.conveyor_items.popleft()
            
            # Find first empty slot to place the item
            try:
                idx = self.inventory.index(None)
                self.inventory[idx] = delivered_item['type']
                # # sfx: Item_Place.wav
                self._create_particles(self._get_slot_center(idx), self.ITEM_COLORS[delivered_item['type']], 10)

                if delivered_item['type'] == 'B':
                    reward -= 0.01 # Penalty for accepting a junk item
            except ValueError:
                # Inventory is full, item is lost
                # # sfx: Item_Discard.wav
                reward -= 0.1 # Penalty for losing an item
        
        return reward

    def _spawn_item(self):
        item_type = 'A' if self.np_random.random() < 0.7 else 'B'
        speed = 2.0 * (self.INITIAL_CONVEYOR_TIMER / self.conveyor_item_interval)
        self.conveyor_items.append({
            'type': item_type,
            'pos_x': self.SCREEN_WIDTH + 50,
            'speed': speed
        })

    def _process_combinations(self):
        reward = 0
        made_combination = True
        
        while made_combination:
            made_combination = False
            for i in range(self.INVENTORY_SIZE - 1):
                item1 = self.inventory[i]
                item2 = self.inventory[i+1]

                if item1 is not None and item1 == item2 and item1 in ['A', 'C']:
                    # # sfx: Combination_Success.wav
                    made_combination = True
                    
                    if item1 == 'A':
                        result = 'C'
                        reward += 1.0 # Create C
                    else: # item1 == 'C'
                        result = 'D'
                        reward += 5.0 # Create D

                    reward += 0.1 # Generic combination reward
                    
                    # Perform combination
                    self.inventory[i] = result
                    self.inventory.pop(i+1)
                    self.inventory.append(None)
                    
                    # Visual/Audio Feedback
                    slot_center = self._get_slot_center(i)
                    self._create_particles(slot_center, self.ITEM_COLORS[result], 30, 5)

                    if result == 'D':
                        self.d_created += 1
                        # # sfx: Goal_Progress.wav
                        # Increase difficulty
                        self.conveyor_item_interval = max(20, int(self.conveyor_item_interval * 0.9))

                    # Restart scan after a combination
                    break 
        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.update()]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw conveyor belt
        belt_y = self.SCREEN_HEIGHT // 2
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, (0, belt_y - 10, self.SCREEN_WIDTH, 20))
        pygame.draw.line(self.screen, self.COLOR_SLOT_HIGHLIGHT, (0, belt_y), (self.SCREEN_WIDTH, belt_y), 1)

        # Draw inventory slots
        slot_size = 80
        slot_spacing = 20
        total_width = self.INVENTORY_SIZE * slot_size + (self.INVENTORY_SIZE - 1) * slot_spacing
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        
        for i in range(self.INVENTORY_SIZE):
            x = start_x + i * (slot_size + slot_spacing)
            y = self.SCREEN_HEIGHT - slot_size - 40
            rect = pygame.Rect(x, y, slot_size, slot_size)
            pygame.draw.rect(self.screen, self.COLOR_SLOT, rect, border_radius=10)
            pygame.draw.rect(self.screen, self.COLOR_SLOT_HIGHLIGHT, rect, width=2, border_radius=10)

        # Draw items in inventory
        for i in range(self.INVENTORY_SIZE):
            item_type = self.inventory[i]
            if item_type:
                center_x, center_y = self._get_slot_center(i)
                self._render_item(self.screen, item_type, (center_x, center_y), 35)

        # Draw items on conveyor
        for item in self.conveyor_items:
            self._render_item(self.screen, item['type'], (int(item['pos_x']), belt_y), 30)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
            
    def _render_item(self, surface, item_type, center_pos, radius):
        color = self.ITEM_COLORS.get(item_type, (255, 255, 255))
        
        # Draw filled circle with antialiasing
        pygame.gfxdraw.filled_circle(surface, center_pos[0], center_pos[1], radius, color)
        # Draw antialiased outline
        pygame.gfxdraw.aacircle(surface, center_pos[0], center_pos[1], radius, self.ITEM_OUTLINE)
        
        # Draw text
        text_surf = self.font_large.render(item_type, True, self.COLOR_BG)
        text_rect = text_surf.get_rect(center=center_pos)
        surface.blit(text_surf, text_rect)

    def _render_ui(self):
        # Render D-Counter
        d_text = f"D-ITEMS: {self.d_created} / {self.TARGET_D_COUNT}"
        d_surf = self.font_medium.render(d_text, True, self.ITEM_COLORS['D'])
        self.screen.blit(d_surf, (20, 20))
        
        # Render Timer
        time_left = (self.MAX_STEPS - self.steps) / 50.0 # Assuming 50 steps/sec for display
        time_text = f"TIME: {time_left:.1f}"
        time_surf = self.font_medium.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(time_surf, time_rect)
        
        # Render game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.d_created >= self.TARGET_D_COUNT:
                msg = "VICTORY!"
                color = self.ITEM_COLORS['D']
            else:
                msg = "TIME UP!"
                color = self.ITEM_COLORS['B']
            
            msg_surf = self.font_large.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "d_created": self.d_created,
            "conveyor_speed_multiplier": self.INITIAL_CONVEYOR_TIMER / self.conveyor_item_interval
        }

    def _get_slot_center(self, index):
        slot_size = 80
        slot_spacing = 20
        total_width = self.INVENTORY_SIZE * slot_size + (self.INVENTORY_SIZE - 1) * slot_spacing
        start_x = (self.SCREEN_WIDTH - total_width) // 2
        
        x = start_x + index * (slot_size + slot_spacing) + slot_size / 2
        y = self.SCREEN_HEIGHT - slot_size / 2 - 40
        return int(x), int(y)

    def _create_particles(self, pos, color, count=20, max_speed=3):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color, max_speed, self.np_random))

    def close(self):
        pygame.quit()

class Particle:
    def __init__(self, x, y, color, max_speed, np_random):
        self.x = x
        self.y = y
        self.color = color
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(1, max_speed)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifespan = np_random.integers(20, 41)
        self.radius = np_random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.radius -= 0.05
        return self.lifespan > 0 and self.radius > 0

    def draw(self, surface):
        if self.radius > 1:
            # Simple square particles are faster to draw than circles
            pygame.draw.rect(surface, self.color, (self.x, self.y, int(self.radius*2), int(self.radius*2)))


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will create a visible window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Conveyor Belt Combiner")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("A or Left Arrow: Shift Left")
    print("D or Right Arrow: Shift Right")
    print("Q: Quit")
    
    while not done:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_a, pygame.K_LEFT]:
                    action[0] = 3 # Left
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    action[0] = 4 # Right
                elif event.key == pygame.K_q:
                    done = True

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, D-Items: {info['d_created']}")
            # Short pause before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(50) # Run at 50 FPS
        
    env.close()