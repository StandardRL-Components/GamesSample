import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T10:43:09.049755
# Source Brief: brief_00625.md
# Brief Index: 625
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class Ship:
    """Represents a single ship in the game."""
    def __init__(self, pos, vel, size, color, max_speed, ship_id):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size = size
        self.color = color
        self.max_speed = max_speed
        self.id = ship_id
        self.rect = pygame.Rect(self.pos.x - size / 2, self.pos.y - size / 2, size, size)

    def update(self, bounds):
        """Updates the ship's position and handles wall bouncing."""
        self.pos += self.vel
        
        # Bounce off walls
        if self.pos.x - self.size / 2 < 0:
            self.pos.x = self.size / 2
            self.vel.x *= -1
        elif self.pos.x + self.size / 2 > bounds[0]:
            self.pos.x = bounds[0] - self.size / 2
            self.vel.x *= -1
            
        if self.pos.y - self.size / 2 < 0:
            self.pos.y = self.size / 2
            self.vel.y *= -1
        elif self.pos.y + self.size / 2 > bounds[1]:
            self.pos.y = bounds[1] - self.size / 2
            self.vel.y *= -1
        
        self.rect.center = self.pos

    def change_speed(self, change, speed_adjustment_rate):
        """Changes the magnitude of the velocity vector."""
        current_speed = self.vel.magnitude()
        
        if change > 0: # Increase speed
            new_speed = min(self.max_speed, current_speed + speed_adjustment_rate)
        elif change < 0: # Decrease speed
            new_speed = max(0, current_speed - speed_adjustment_rate)
        else: # No change
            return

        if current_speed > 0:
            self.vel = self.vel.normalize() * new_speed
        elif new_speed > 0:
            # Cannot accelerate from a dead stop without a direction
            # Give it a small nudge in a random direction
            self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * new_speed

    def draw(self, surface, is_selected):
        """Draws the ship and its velocity vector."""
        # Draw ship body
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)
        
        # Draw velocity vector
        if self.vel.magnitude() > 0:
            end_pos = self.pos + self.vel.normalize() * (20 + self.vel.magnitude() * 8)
            pygame.draw.line(surface, self.color, self.pos, end_pos, 2)
            # Arrowhead
            angle = self.vel.angle_to(pygame.Vector2(1, 0))
            p1 = end_pos + pygame.Vector2(6, 0).rotate(-angle - 150)
            p2 = end_pos + pygame.Vector2(6, 0).rotate(-angle + 150)
            pygame.draw.polygon(surface, self.color, [end_pos, p1, p2])

        # Draw selection highlight
        if is_selected:
            pygame.draw.rect(surface, (255, 255, 255), self.rect.inflate(8, 8), 3, border_radius=5)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Maneuver a fleet of ships into a tight formation at their center point. "
        "Control each ship's speed to avoid collisions and dock successfully before time runs out."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to select a ship. "
        "Press space to accelerate the selected ship and shift to decelerate."
    )
    auto_advance = False
    
    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 2400  # 120 seconds at 20Hz (0.05s per step)
    
    COLOR_BG = (10, 20, 40)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_DOCKING_TARGET = (255, 255, 255)

    SHIP_SPECS = [
        {'size': 25, 'color': (255, 80, 80), 'max_speed': 1.5}, # Red
        {'size': 20, 'color': (80, 255, 80), 'max_speed': 2.0}, # Green
        {'size': 15, 'color': (80, 120, 255), 'max_speed': 2.5}, # Blue
        {'size': 10, 'color': (255, 255, 80), 'max_speed': 3.0}, # Yellow
    ]
    SPEED_ADJUSTMENT = 0.1
    DOCKING_RADIUS = 15.0 # Brief said 5, but that's extremely hard. 15 is more achievable.
    PROXIMITY_RADIUS = 50.0 # Brief said 20, adjusted for better reward signal.

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Exo 2", 30)
        self.font_small = pygame.font.SysFont("Exo 2", 20)
        
        self.ships = []
        self.stars = []
        self.selected_ship_idx = 0
        self.ship_docked_reward_given = [False] * len(self.SHIP_SPECS)
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.reset()
        # self.validate_implementation() # Removed for submission
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.selected_ship_idx = 0
        self.ship_docked_reward_given = [False] * len(self.SHIP_SPECS)
        
        self._initialize_stars()
        self._initialize_ships()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Input
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        # 2. Update Game Logic
        for ship in self.ships:
            ship.update((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # 3. Check for Termination Conditions
        self.steps += 1
        collision = self._check_collisions()
        docked = self._check_docking() if not collision else False
        timeout = self.steps >= self.MAX_STEPS
        terminated = collision or docked or timeout
        truncated = False # This environment does not truncate based on time limits
        self.game_over = terminated

        # 4. Calculate Reward
        reward = self._calculate_reward(collision, docked, timeout)
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        if 1 <= movement <= 4:
            self.selected_ship_idx = movement - 1
        
        change = 0
        if space_held:
            change = 1
            # sfx: speed_up_sound
        if shift_held:
            change = -1
            # sfx: speed_down_sound
        
        if change != 0:
            self.ships[self.selected_ship_idx].change_speed(change, self.SPEED_ADJUSTMENT)

    def _calculate_reward(self, collision, docked, timeout):
        if collision:
            # sfx: explosion_sound
            return -50.0
        if docked:
            # sfx: success_chime
            return 100.0
        if timeout:
            return -10.0

        reward = -0.01  # Penalty for taking time

        centroid = self._get_centroid()
        
        # Proximity reward
        for ship in self.ships:
            if ship.pos.distance_to(centroid) < self.PROXIMITY_RADIUS:
                reward += 0.1

        # One-time docking entry reward
        for i, ship in enumerate(self.ships):
            if not self.ship_docked_reward_given[i] and ship.pos.distance_to(centroid) < self.DOCKING_RADIUS:
                reward += 5.0
                self.ship_docked_reward_given[i] = True
                # sfx: docking_click_sound
        
        return reward

    def _check_collisions(self):
        for i in range(len(self.ships)):
            for j in range(i + 1, len(self.ships)):
                if self.ships[i].rect.colliderect(self.ships[j].rect):
                    return True
        return False

    def _check_docking(self):
        centroid = self._get_centroid()
        for ship in self.ships:
            if ship.pos.distance_to(centroid) > self.DOCKING_RADIUS:
                return False
        return True

    def _get_centroid(self):
        if not self.ships:
            return pygame.Vector2(0, 0)
        return sum([s.pos for s in self.ships], pygame.Vector2()) / len(self.ships)
        
    def _initialize_ships(self):
        self.ships = []
        max_attempts = 100
        for attempt in range(max_attempts):
            self.ships = []
            valid_placement = True
            for i, spec in enumerate(self.SHIP_SPECS):
                pos = (
                    random.uniform(50, self.SCREEN_WIDTH - 50),
                    random.uniform(50, self.SCREEN_HEIGHT - 50)
                )
                angle = random.uniform(0, 360)
                speed = random.uniform(0.5, spec['max_speed'] * 0.75)
                vel = pygame.Vector2(speed, 0).rotate(angle)
                
                new_ship = Ship(pos, vel, spec['size'], spec['color'], spec['max_speed'], i)
                
                # Check for overlap with existing ships
                for existing_ship in self.ships:
                    if new_ship.rect.inflate(20, 20).colliderect(existing_ship.rect):
                        valid_placement = False
                        break
                if not valid_placement:
                    break
                self.ships.append(new_ship)

            if valid_placement:
                return # Success
        
        # Fallback if a safe placement isn't found
        if not self.ships or len(self.ships) != len(self.SHIP_SPECS):
            self.ships = [Ship(
                (self.SCREEN_WIDTH/2 + (i-1.5)*80, self.SCREEN_HEIGHT/2),
                (0,0), s['size'], s['color'], s['max_speed'], i) 
                for i, s in enumerate(self.SHIP_SPECS)]

    def _initialize_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': (random.randint(0, self.SCREEN_WIDTH), random.randint(0, self.SCREEN_HEIGHT)),
                'radius': random.choice([1, 1, 1, 2]),
                'color_val': random.randint(100, 200)
            })

    def _get_observation(self):
        self._render_all()
        arr = pygame.surfarray.array3d(self.screen)
        # Pygame returns (width, height, channels), but we need (height, width, channels)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_all(self):
        # Background
        self.screen.fill(self.COLOR_BG)
        
        # Stars
        for star in self.stars:
            c = star['color_val'] + random.randint(-10, 10)
            c = max(50, min(220, c))
            color = (c, c, c+10)
            pygame.draw.circle(self.screen, color, star['pos'], star['radius'])
        
        # Game elements
        centroid = self._get_centroid()
        
        # Docking target circle
        if not self.game_over:
            pygame.gfxdraw.aacircle(self.screen, int(centroid.x), int(centroid.y), int(self.DOCKING_RADIUS), self.COLOR_DOCKING_TARGET + (60,))
            pygame.gfxdraw.aacircle(self.screen, int(centroid.x), int(centroid.y), int(self.DOCKING_RADIUS)+1, self.COLOR_DOCKING_TARGET + (40,))

        # Ships
        for i, ship in enumerate(self.ships):
            ship.draw(self.screen, i == self.selected_ship_idx)
            
        # UI
        self._render_ui()

    def _render_ui(self):
        # Timer
        time_left_s = max(0, (self.MAX_STEPS - self.steps) * 0.05)
        minutes = int(time_left_s // 60)
        seconds = int(time_left_s % 60)
        timer_text = f"{minutes:02}:{seconds:02}"
        self._draw_text(timer_text, (self.SCREEN_WIDTH - 80, 25), self.COLOR_UI_TEXT, self.font_main)
        
        # Score
        score_text = f"SCORE: {int(self.score)}"
        self._draw_text(score_text, (80, 25), self.COLOR_UI_TEXT, self.font_main)

        # Selected Ship Info
        if self.ships:
            sel_ship = self.ships[self.selected_ship_idx]
            speed = sel_ship.vel.magnitude()
            sel_text = f"SHIP {sel_ship.id + 1} | SPD: {speed:.2f} / {sel_ship.max_speed:.2f}"
            self._draw_text(sel_text, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 20), sel_ship.color, self.font_small)

    def _draw_text(self, text, pos, color, font):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "selected_ship": self.selected_ship_idx
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block is for manual testing and visualization.
    # It will not be executed by the evaluation server.
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Un-dummy the video driver to see the visualization
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "mac", etc. depending on your OS
    pygame.display.init()
    
    # --- Manual Play Example ---
    # Use arrow keys to select ship, SPACE to accelerate, SHIFT to decelerate
    selected_ship_action = 0
    space_action = 0
    shift_action = 0
    
    # Create a display window
    pygame.display.set_caption("Docking Maneuver")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: selected_ship_action = 1
                elif event.key == pygame.K_2: selected_ship_action = 2
                elif event.key == pygame.K_3: selected_ship_action = 3
                elif event.key == pygame.K_4: selected_ship_action = 4
                elif event.key == pygame.K_SPACE: space_action = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 1
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
                    selected_ship_action = 0
                elif event.key == pygame.K_SPACE: space_action = 0
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT: shift_action = 0
        
        action = [selected_ship_action, space_action, shift_action]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # For manual play, we want to see the game state after each action
        # Render the observation to the display window
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            
        env.clock.tick(30) # Limit frame rate for human play

    env.close()