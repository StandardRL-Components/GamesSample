import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:27:15.944444
# Source Brief: brief_00995.md
# Brief Index: 995
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment where the player controls two platforms to catch falling items.

    **Gameplay:**
    - Control two platforms at the bottom of the screen independently.
    - Green squares (items) fall from the top. Catch them with the correct platform to score points.
    - Red circles (bombs) also fall. Catching them triggers a chain reaction, destroying nearby
      items and penalizing the player.
    - The goal is to collect 10 items on each platform before the 90-second timer runs out.

    **Action Space:** `MultiDiscrete([5, 2, 2])`
    - `action[0]`: Controls the Left Platform (3=left, 4=right, 0/1/2=none).
    - `action[1]`: If held (1), moves the Right Platform to the left.
    - `action[2]`: If held (1), moves the Right Platform to the right.
      (If both are held, the Right Platform does not move).

    **Observation Space:** `Box(400, 640, 3)`
    - An RGB image of the game screen.

    **Rewards:**
    - +1 for catching a green item.
    - -5 for catching a red bomb.
    - +10 for reaching the 10-item goal on a single platform.
    - +100 for winning the game (10 items on both platforms).
    - -100 for losing (timer runs out or it becomes impossible to win).
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Control two platforms to catch falling items and avoid bombs. "
        "The goal is to collect 10 items on each platform before the timer runs out."
    )
    user_guide = (
        "Use A/D to move the left platform and the ←/→ arrow keys to move the right platform. "
        "Catch green items and avoid red bombs."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    MAX_TIME_SECONDS = 90

    # Colors
    COLOR_BG = (15, 20, 30)
    COLOR_PLATFORM_L = (0, 150, 255)
    COLOR_PLATFORM_R = (0, 200, 200)
    COLOR_PLATFORM_L_GLOW = (100, 200, 255)
    COLOR_PLATFORM_R_GLOW = (100, 225, 225)
    COLOR_ITEM = (50, 220, 120)
    COLOR_BOMB = (255, 80, 80)
    COLOR_BOMB_GLOW = (255, 150, 150)
    COLOR_UI_TEXT = (255, 220, 100)
    COLOR_PARTICLE_DESTROY = (100, 100, 110)

    # Game Parameters
    PLATFORM_WIDTH = 100
    PLATFORM_HEIGHT = 15
    PLATFORM_Y = SCREEN_HEIGHT - 30
    PLATFORM_SPEED = 8.0
    ITEM_SIZE = 16
    BOMB_RADIUS = 10
    INITIAL_FALL_SPEED = 2.0
    MAX_FALL_SPEED = 5.0
    FALL_SPEED_INCREASE = 0.05
    SPEED_INCREASE_INTERVAL = 300 # frames
    ITEM_SPAWN_RATE = 25 # frames
    BOMB_PROBABILITY = 0.15
    WIN_SCORE = 10
    CHAIN_REACTION_RADIUS = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium Spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)

        # Game State Variables (initialized in reset)
        self.steps = 0
        self.timer = 0
        self.score_left = 0
        self.score_right = 0
        self.game_over = False
        self.win_status = False

        self.platform_left_x = 0
        self.platform_right_x = 0
        self.items = []
        self.particles = []
        
        self.item_spawn_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.timer = self.MAX_TIME_SECONDS * self.FPS
        self.score_left = 0
        self.score_right = 0
        self.game_over = False
        self.win_status = False

        self.platform_left_x = self.SCREEN_WIDTH * 0.25 - self.PLATFORM_WIDTH / 2
        self.platform_right_x = self.SCREEN_WIDTH * 0.75 - self.PLATFORM_WIDTH / 2

        self.items = []
        self.particles = []
        self.item_spawn_timer = 0
        self.fall_speed = self.INITIAL_FALL_SPEED

        # Spawn a few initial items to start the game
        for _ in range(5):
            self._spawn_item()
            # Stagger their initial positions
            if self.items:
                 self.items[-1]['pos'][1] -= self.np_random.uniform(50, 350)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1
        reward = 0

        # --- Update Game Logic ---
        self._update_platforms(action)
        self._update_items()
        self._update_particles()
        self._handle_spawning()
        self._update_difficulty()
        
        # --- Handle Collisions and Rewards ---
        reward += self._handle_collisions()

        # --- Check for Termination ---
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if self.win_status:
                reward += 100 # Win bonus
                # sfx: win_sound
            else:
                reward -= 100 # Lose penalty
                # sfx: lose_sound
        
        # Truncated is always False as per the brief
        truncated = False

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_platforms(self, action):
        # Unpack factorized action
        left_platform_action = action[0]
        right_platform_move_left = action[1] == 1
        right_platform_move_right = action[2] == 1

        # Left platform movement
        if left_platform_action == 3: # Left
            self.platform_left_x -= self.PLATFORM_SPEED
        elif left_platform_action == 4: # Right
            self.platform_left_x += self.PLATFORM_SPEED
        
        # Right platform movement
        right_velocity = 0
        if right_platform_move_left:
            right_velocity -= self.PLATFORM_SPEED
        if right_platform_move_right:
            right_velocity += self.PLATFORM_SPEED
        self.platform_right_x += right_velocity

        # Clamp platform positions to screen bounds
        self.platform_left_x = max(0, min(self.platform_left_x, self.SCREEN_WIDTH / 2 - self.PLATFORM_WIDTH))
        self.platform_right_x = max(self.SCREEN_WIDTH / 2, min(self.platform_right_x, self.SCREEN_WIDTH - self.PLATFORM_WIDTH))

    def _update_items(self):
        for item in self.items:
            item['pos'][1] += self.fall_speed
        # Remove items that fall off the bottom of the screen
        self.items = [item for item in self.items if item['pos'][1] < self.SCREEN_HEIGHT]

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _handle_spawning(self):
        self.item_spawn_timer -= 1
        if self.item_spawn_timer <= 0:
            self._spawn_item()
            self.item_spawn_timer = self.ITEM_SPAWN_RATE

    def _update_difficulty(self):
        if self.steps > 0 and self.steps % self.SPEED_INCREASE_INTERVAL == 0:
            self.fall_speed = min(self.MAX_FALL_SPEED, self.fall_speed + self.FALL_SPEED_INCREASE)

    def _spawn_item(self):
        item_type = 'bomb' if self.np_random.random() < self.BOMB_PROBABILITY else 'item'
        x_pos = self.np_random.uniform(self.ITEM_SIZE, self.SCREEN_WIDTH - self.ITEM_SIZE)
        
        self.items.append({
            'pos': [x_pos, -self.ITEM_SIZE],
            'type': item_type
        })
    
    def _handle_collisions(self):
        step_reward = 0
        platform_left_rect = pygame.Rect(self.platform_left_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        platform_right_rect = pygame.Rect(self.platform_right_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)

        items_to_remove = []
        for i, item in enumerate(self.items):
            item_rect = pygame.Rect(item['pos'][0] - self.ITEM_SIZE/2, item['pos'][1] - self.ITEM_SIZE/2, self.ITEM_SIZE, self.ITEM_SIZE)

            # Left Platform Collision
            if platform_left_rect.colliderect(item_rect):
                if item['type'] == 'item':
                    self.score_left += 1
                    step_reward += 1
                    if self.score_left == self.WIN_SCORE:
                        step_reward += 10 # Goal bonus
                    self._create_particles(item['pos'], self.COLOR_ITEM, 20)
                    # sfx: collect_item
                else: # Bomb
                    step_reward -= 5
                    self._trigger_chain_reaction(item['pos'])
                    # sfx: explosion
                items_to_remove.append(i)

            # Right Platform Collision
            elif platform_right_rect.colliderect(item_rect):
                if item['type'] == 'item':
                    self.score_right += 1
                    step_reward += 1
                    if self.score_right == self.WIN_SCORE:
                        step_reward += 10 # Goal bonus
                    self._create_particles(item['pos'], self.COLOR_ITEM, 20)
                    # sfx: collect_item
                else: # Bomb
                    step_reward -= 5
                    self._trigger_chain_reaction(item['pos'])
                    # sfx: explosion
                items_to_remove.append(i)

        if items_to_remove:
            self.items = [item for i, item in enumerate(self.items) if i not in items_to_remove]
        
        return step_reward

    def _trigger_chain_reaction(self, pos):
        self._create_particles(pos, self.COLOR_BOMB, 50, speed_multiplier=2.0)
        
        destroyed_in_chain = []
        for i, item in enumerate(self.items):
            distance = math.hypot(item['pos'][0] - pos[0], item['pos'][1] - pos[1])
            if distance < self.CHAIN_REACTION_RADIUS:
                destroyed_in_chain.append(i)
                self._create_particles(item['pos'], self.COLOR_PARTICLE_DESTROY, 10, speed_multiplier=0.5)

        if destroyed_in_chain:
            self.items = [item for i, item in enumerate(self.items) if i not in destroyed_in_chain]

    def _check_termination(self):
        # Win condition
        if self.score_left >= self.WIN_SCORE and self.score_right >= self.WIN_SCORE:
            self.win_status = True
            return True
        
        # Lose condition: timer runs out
        if self.timer <= 0:
            return True
            
        # Lose condition: impossible to win
        items_on_screen = sum(1 for item in self.items if item['type'] == 'item')
        needed_left = max(0, self.WIN_SCORE - self.score_left)
        needed_right = max(0, self.WIN_SCORE - self.score_right)
        
        # A simple proxy: if no items are on screen and more are needed, it's a loss.
        # A more complex check would involve estimating future spawns, which is out of scope.
        if items_on_screen == 0 and (needed_left > 0 or needed_right > 0) and self.item_spawn_timer > 0:
             # Check if there are no items and none will spawn soon enough
             if self.timer / self.ITEM_SPAWN_RATE < max(needed_left, needed_right):
                 return True

        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score_left": self.score_left,
            "score_right": self.score_right,
            "steps": self.steps,
            "timer": self.timer,
            "fall_speed": self.fall_speed
        }

    def _render_game(self):
        # Draw Particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['lifespan'] / p['max_lifespan']))))
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))


        # Draw Items
        for item in self.items:
            pos = (int(item['pos'][0]), int(item['pos'][1]))
            if item['type'] == 'item':
                rect = pygame.Rect(pos[0] - self.ITEM_SIZE / 2, pos[1] - self.ITEM_SIZE / 2, self.ITEM_SIZE, self.ITEM_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_ITEM, rect, border_radius=3)
            else: # Bomb
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.COLOR_BOMB_GLOW)
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.BOMB_RADIUS, self.COLOR_BOMB)

        # Draw Platforms with glow effect
        # Left Platform
        platform_l_rect = pygame.Rect(self.platform_left_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        glow_l_rect = platform_l_rect.inflate(6, 6)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_L_GLOW, glow_l_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_L, platform_l_rect, border_radius=6)
        
        # Right Platform
        platform_r_rect = pygame.Rect(self.platform_right_x, self.PLATFORM_Y, self.PLATFORM_WIDTH, self.PLATFORM_HEIGHT)
        glow_r_rect = platform_r_rect.inflate(6, 6)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_R_GLOW, glow_r_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PLATFORM_R, platform_r_rect, border_radius=6)

    def _render_ui(self):
        # Draw Timer
        time_left_sec = self.timer / self.FPS
        timer_text = f"{max(0, time_left_sec):.1f}"
        text_surface = self.font_large.render(timer_text, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH / 2, 30))
        self.screen.blit(text_surface, text_rect)

        # Draw Scores
        # Left Score
        left_score_text = f"{self.score_left}/{self.WIN_SCORE}"
        left_text_surface = self.font_small.render(left_score_text, True, self.COLOR_UI_TEXT)
        left_text_rect = left_text_surface.get_rect(center=(self.platform_left_x + self.PLATFORM_WIDTH / 2, self.PLATFORM_Y - 20))
        self.screen.blit(left_text_surface, left_text_rect)

        # Right Score
        right_score_text = f"{self.score_right}/{self.WIN_SCORE}"
        right_text_surface = self.font_small.render(right_score_text, True, self.COLOR_UI_TEXT)
        right_text_rect = right_text_surface.get_rect(center=(self.platform_right_x + self.PLATFORM_WIDTH / 2, self.PLATFORM_Y - 20))
        self.screen.blit(right_text_surface, right_text_rect)

    def _create_particles(self, pos, color, count, speed_multiplier=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_multiplier
            lifespan = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'lifespan': lifespan,
                'max_lifespan': lifespan,
                'color': color,
                'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.font.quit()
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run in the headless environment.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Dual Catcher")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0

    while not terminated:
        # --- Human Controls ---
        keys = pygame.key.get_pressed()
        
        # Left platform
        left_action = 0 # no-op
        if keys[pygame.K_a]:
            left_action = 3 # left
        elif keys[pygame.K_d]:
            left_action = 4 # right

        # Right platform
        right_move_left = 1 if keys[pygame.K_LEFT] else 0
        right_move_right = 1 if keys[pygame.K_RIGHT] else 0
        
        action = [left_action, right_move_left, right_move_right]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score (sum of rewards): {total_reward}")
    print(f"Info: {info}")
    
    env.close()