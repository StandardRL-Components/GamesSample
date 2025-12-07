import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:58:13.871702
# Source Brief: brief_01436.md
# Brief Index: 1436
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player sorts colored items on two conveyor belts
    into their matching chutes. The player controls the direction of each belt.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Sort colored items into their matching chutes by controlling the direction of two conveyor belts. "
        "Work quickly before time runs out!"
    )
    user_guide = (
        "Controls: Use ↑/↓ to direct the top belt, and ←/→ to direct the bottom belt."
    )
    auto_advance = True


    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Game Parameters
    TOTAL_ITEMS = 20
    MAX_TIME_SECONDS = 60
    MAX_STEPS = MAX_TIME_SECONDS * FPS # Safety termination
    BELT_SPEED_INITIAL = 2.0
    BELT_SPEED_INCREMENT = 0.5
    ITEMS_PER_SPEED_INCREASE = 5
    JAM_DURATION_FRAMES = 1 * FPS # 1 second jam

    # Colors (Industrial & Clean)
    COLOR_BG = (25, 35, 45)
    COLOR_BELT = (60, 70, 80)
    COLOR_BELT_SHADOW = (40, 50, 60)
    COLOR_TEXT = (220, 220, 220)
    COLOR_JAM = (255, 50, 50)
    ITEM_COLORS = [
        (50, 150, 255),  # Blue
        (50, 255, 150),  # Green
        (255, 200, 50),  # Yellow
        (255, 100, 100), # Red
        (200, 100, 255)  # Purple
    ]
    NUM_ITEM_TYPES = len(ITEM_COLORS)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        # Action: [Belt Control, Unused, Unused]
        # Belt Control: 0=NoOp, 1=TopL, 2=TopR, 3=BotL, 4=BotR
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 36, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 24)
        self.font_small = pygame.font.SysFont("Consolas", 16)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.time_remaining = 0
        
        self.belt1_direction = 1  # 1 for right, -1 for left
        self.belt2_direction = 1
        self.belt_speed = self.BELT_SPEED_INITIAL
        
        self.sorted_count = 0
        self.items_to_spawn = []
        self.active_items = []
        self.particles = []
        self.spawn_timer = 0
        
        # --- Pre-calculated positions for rendering ---
        self.belt_height = 60
        self.belt_y_1 = self.SCREEN_HEIGHT // 2 - self.belt_height - 15
        self.belt_y_2 = self.SCREEN_HEIGHT // 2 + 15
        self.chute_width = 40
        self.item_size = 30

        # Note: self.reset() is not called in __init__ per Gymnasium standard practice.
        # The environment is expected to be reset by the user.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.time_remaining = self.MAX_TIME_SECONDS * self.FPS
        
        self.belt1_direction = 1
        self.belt2_direction = 1
        self.belt_speed = self.BELT_SPEED_INITIAL
        
        self.sorted_count = 0
        self.items_to_spawn = [self.np_random.integers(0, self.NUM_ITEM_TYPES) for _ in range(self.TOTAL_ITEMS)]
        self.active_items = []
        self.particles = []
        
        self.spawn_timer = self.FPS // 2 # Start spawning quickly

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- 1. Handle Input ---
        movement = action[0]
        # Actions 1-2 (up/down) control top belt
        if movement == 1: self.belt1_direction = -1 # Left
        elif movement == 2: self.belt1_direction = 1  # Right
        # Actions 3-4 (left/right) control bottom belt
        elif movement == 3: self.belt2_direction = -1 # Left
        elif movement == 4: self.belt2_direction = 1  # Right
        # Action 0 is no-op

        # --- 2. Update Game Logic ---
        self.steps += 1
        self.time_remaining -= 1
        reward = 0

        # Update belt speed based on progress
        self.belt_speed = self.BELT_SPEED_INITIAL + (self.sorted_count // self.ITEMS_PER_SPEED_INCREASE) * self.BELT_SPEED_INCREMENT

        # Update particles
        self._update_particles()

        # Spawn new items
        self._update_spawner()

        # Update items on belts
        reward += self._update_items()

        # --- 3. Check Termination ---
        terminated = False
        truncated = False
        if self.sorted_count >= self.TOTAL_ITEMS:
            self.win_condition = True
            terminated = True
            reward += 100 # Win bonus
        elif self.time_remaining <= 0:
            terminated = True
            reward -= 100 # Lose penalty
        
        if self.steps >= self.MAX_STEPS:
            truncated = True

        if terminated or truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_spawner(self):
        self.spawn_timer -= 1
        if self.spawn_timer <= 0 and self.items_to_spawn:
            item_type = self.items_to_spawn.pop(0)
            belt_idx = self.np_random.integers(0, 2)
            y_pos = self.belt_y_1 + (self.belt_height - self.item_size) / 2 if belt_idx == 0 else self.belt_y_2 + (self.belt_height - self.item_size) / 2
            
            new_item = {
                'pos': [self.SCREEN_WIDTH / 2, y_pos],
                'color_idx': item_type,
                'belt_idx': belt_idx,
                'jam_timer': 0
            }
            self.active_items.append(new_item)
            self.spawn_timer = self.np_random.integers(int(1.5 * self.FPS), int(3.0 * self.FPS))

    def _update_items(self):
        step_reward = 0
        items_to_remove = []
        for item in self.active_items:
            # Handle jams
            if item['jam_timer'] > 0:
                item['jam_timer'] -= 1
                step_reward -= 1.0 / self.FPS # Penalty for being jammed
                if item['jam_timer'] <= 0:
                    items_to_remove.append(item) # Jam cleared, remove item
                continue

            # Move item
            belt_dir = self.belt1_direction if item['belt_idx'] == 0 else self.belt2_direction
            item['pos'][0] += belt_dir * self.belt_speed

            # Continuous reward for correct direction
            # The correct chute is on the right for type 0, 1 and on the left for type 2, 3, 4
            correct_dir = 1 if item['color_idx'] < self.NUM_ITEM_TYPES / 2 else -1
            if belt_dir == correct_dir:
                step_reward += 0.01 # Small reward for moving correctly
            else:
                step_reward -= 0.02 # Small penalty for moving incorrectly

            # Check for sorting/jamming at belt ends
            item_x = item['pos'][0]
            if item_x < self.chute_width or item_x > self.SCREEN_WIDTH - self.chute_width - self.item_size:
                is_left_end = item_x < self.chute_width
                target_dir = -1 if is_left_end else 1
                
                if target_dir == correct_dir:
                    # Correct sort
                    self.score += 10
                    self.sorted_count += 1
                    step_reward += 1
                    items_to_remove.append(item)
                    self._create_particles(item['pos'], self.ITEM_COLORS[item['color_idx']])
                else:
                    # Incorrect sort -> Jam
                    step_reward -= 2
                    item['jam_timer'] = self.JAM_DURATION_FRAMES
                    # Snap item to edge to make jam visible
                    item['pos'][0] = 0 if is_left_end else self.SCREEN_WIDTH - self.item_size

        # Clean up processed items
        self.active_items = [item for item in self.active_items if item not in items_to_remove]
        return step_reward

    def _create_particles(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                'pos': [pos[0] + self.item_size / 2, pos[1] + self.item_size / 2],
                'vel': vel,
                'color': color,
                'lifetime': self.np_random.integers(20, 41),
                'size': self.np_random.uniform(2, 5)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifetime'] -= 1
            p['size'] -= 0.05
        self.particles = [p for p in self.particles if p['lifetime'] > 0 and p['size'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Belts and Chutes ---
        belt_rect_1 = pygame.Rect(0, self.belt_y_1, self.SCREEN_WIDTH, self.belt_height)
        belt_rect_2 = pygame.Rect(0, self.belt_y_2, self.SCREEN_WIDTH, self.belt_height)
        
        # Shadow
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, belt_rect_1.move(0, 5))
        pygame.draw.rect(self.screen, self.COLOR_BELT_SHADOW, belt_rect_2.move(0, 5))
        
        # Main belt
        pygame.draw.rect(self.screen, self.COLOR_BELT, belt_rect_1)
        pygame.draw.rect(self.screen, self.COLOR_BELT, belt_rect_2)

        # Chutes (Left side for types 2,3,4, Right side for 0,1)
        for i in range(self.NUM_ITEM_TYPES):
            chute_color = self.ITEM_COLORS[i]
            if i < self.NUM_ITEM_TYPES / 2: # Right side
                pygame.draw.rect(self.screen, chute_color, (self.SCREEN_WIDTH - self.chute_width, 0, self.chute_width, self.SCREEN_HEIGHT))
            else: # Left side
                pygame.draw.rect(self.screen, chute_color, (0, 0, self.chute_width, self.SCREEN_HEIGHT))

        # --- Draw Direction Indicators ---
        self._draw_arrow(self.belt_y_1 + self.belt_height / 2, self.belt1_direction)
        self._draw_arrow(self.belt_y_2 + self.belt_height / 2, self.belt2_direction)
        
        # --- Draw Items ---
        for item in self.active_items:
            item_rect = pygame.Rect(int(item['pos'][0]), int(item['pos'][1]), self.item_size, self.item_size)
            item_color = self.ITEM_COLORS[item['color_idx']]
            
            # Jam effect
            if item['jam_timer'] > 0 and self.steps % 10 < 5:
                jam_rect = item_rect.inflate(10, 10)
                pygame.draw.rect(self.screen, self.COLOR_JAM, jam_rect, border_radius=5)

            # Item body with highlight
            pygame.draw.rect(self.screen, tuple(min(255, c + 40) for c in item_color), item_rect.move(-2, -2), border_radius=4)
            pygame.draw.rect(self.screen, item_color, item_rect, border_radius=4)
            
        # --- Draw Particles ---
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            # pygame.gfxdraw requires alpha channel in the color tuple
            alpha = max(0, min(255, int(255 * (p['lifetime'] / 40.0))))
            color_with_alpha = p['color'] + (alpha,)
            try:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color_with_alpha)
            except TypeError: # Sometimes color doesn't have alpha, ensure it does
                if len(color_with_alpha) != 4:
                     color_with_alpha = p['color'] + (alpha,)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['size']), color_with_alpha)

    def _draw_arrow(self, y_center, direction):
        color = (100, 255, 100)
        x_center = self.SCREEN_WIDTH / 2
        if direction == 1: # Right
            points = [(x_center - 10, y_center - 10), (x_center + 10, y_center), (x_center - 10, y_center + 10)]
        else: # Left
            points = [(x_center + 10, y_center - 10), (x_center - 10, y_center), (x_center + 10, y_center + 10)]
        pygame.draw.polygon(self.screen, color, points)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_str = f"{max(0, self.time_remaining // self.FPS):02d}"
        time_text = self.font_large.render(f"TIME: {time_str}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 20, 10))
        
        # Items sorted
        sorted_text = self.font_medium.render(f"SORTED: {self.sorted_count}/{self.TOTAL_ITEMS}", True, self.COLOR_TEXT)
        self.screen.blit(sorted_text, (20, 50))

        # Belt Speed
        speed_text = self.font_small.render(f"BELT SPEED: {self.belt_speed:.1f}x", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH/2 - speed_text.get_width()/2, self.SCREEN_HEIGHT - 30))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "MISSION COMPLETE" if self.win_condition else "TIME UP"
            color = (100, 255, 100) if self.win_condition else self.COLOR_JAM
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "sorted_count": self.sorted_count,
            "time_remaining_seconds": max(0, self.time_remaining // self.FPS),
            "belt_speed": self.belt_speed,
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv()
    
    # --- Manual Play ---
    # Controls: 1/2 for top belt (L/R), 3/4 for bottom belt (L/R)
    # W/S for top belt, A/D for bottom belt for easier human play
    
    obs, info = env.reset()
    done = False
    
    # Pygame window for human play
    os.environ["SDL_VIDEODRIVER"] = "x11" # or "windows", "mac", etc.
    pygame.display.init()
    pygame.display.set_caption("Dual Conveyor Sorter")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0])
    
    while not done:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            # Reset action on keyup to prevent sticky controls
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    action[0] = 0

            # Set action on keydown
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action = np.array([0, 0, 0])
                    continue
                if event.key == pygame.K_w: # Top belt left
                    action[0] = 1
                if event.key == pygame.K_s: # Top belt right
                    action[0] = 2
                if event.key == pygame.K_a: # Bottom belt left
                    action[0] = 3
                if event.key == pygame.K_d: # Bottom belt right
                    action[0] = 4

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()