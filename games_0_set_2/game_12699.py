import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:17:48.433311
# Source Brief: brief_02699.md
# Brief Index: 2699
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Sort the numbered balls on the conveyor belt into ascending order. "
        "Swap adjacent balls or shift a ball to the end of the line to solve the puzzle within the turn limit."
    )
    user_guide = (
        "Controls: Use ←→ to move the selector. Hold Shift and press ←→ to swap adjacent balls. "
        "Press Space to move the selected ball to the end of the line."
    )
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    NUM_BALLS = 12
    MAX_TURNS = 20

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_CONVEYOR = (40, 50, 60)
    COLOR_CONVEYOR_BORDER = (60, 70, 80)
    COLOR_TEXT = (220, 220, 230)
    COLOR_SCORE = (255, 200, 0)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)
    BALL_COLORS = [
        (255, 87, 87),    # 1 (Red)
        (255, 170, 87),   # 2 (Orange)
        (255, 255, 87),   # 3 (Yellow)
        (170, 255, 87),   # 4 (Lime)
        (87, 255, 87),    # 5 (Green)
        (87, 255, 170),   # 6 (Teal)
        (87, 255, 255),   # 7 (Cyan)
        (87, 170, 255),   # 8 (Sky Blue)
        (87, 87, 255),    # 9 (Blue)
        (170, 87, 255),   # 10 (Purple)
        (255, 87, 255),   # 11 (Magenta)
        (255, 87, 170),   # 12 (Pink)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_ball = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_endgame = pygame.font.SysFont("Arial", 64, bold=True)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turn_count = 0
        self.balls = []
        self.selected_index = 0
        self.is_animating = False
        self.animation_queue = []
        self.particles = []
        self.win_state = None
        self.ball_visual_pos = []
        self.selector_visual_pos = pygame.Vector2(0, 0)

        # self.reset() is called by the wrapper, but we can call it here for standalone use
        # self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.turn_count = 0
        
        self.balls = list(range(1, self.NUM_BALLS + 1))
        # In Gymnasium, self.np_random is a np.random.Generator
        self.np_random.shuffle(self.balls)

        self.selected_index = 0
        self.win_state = None
        
        self.is_animating = False
        self.animation_queue.clear()
        self.particles.clear()
        
        self.ball_visual_pos = [self._get_ball_target_pos(i) for i in range(self.NUM_BALLS)]
        self.selector_visual_pos = self._get_ball_target_pos(self.selected_index)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        # --- Update animations and game logic ---
        # In a non-auto_advance game, we may want to step animations forward.
        # Here we'll process one frame of animation per step.
        self._update_animations()

        # Only process new actions if no animation is running
        if not self.is_animating:
            action_taken = self._handle_action(action)
            
            # If an action was taken, it starts an animation.
            # The end-of-turn logic will be triggered when the animation finishes.
            if action_taken:
                # Immediate reward for the action itself
                if 'type' in action_taken:
                    if action_taken['type'] == 'swap':
                        i, j = action_taken['indices']
                        # Check if swap created a correct adjacency
                        if (self.balls[i] == self.balls[j] - 1) or (self.balls[j] == self.balls[i] - 1):
                            reward += 5
                            # SFX: Positive clink
                            self._spawn_particles(self._get_ball_target_pos(j), self.COLOR_SCORE, 15)

        # We can loop animations until they are done since auto_advance is False
        while self.is_animating:
            self._update_animations()

        self.score += reward
        terminated = self.game_over
        truncated = False # No truncation condition in this game

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        action_details = None

        # --- Action: Perform Shift (Spacebar) ---
        if space_held:
            # SFX: Whoosh
            self._start_turn()
            
            val = self.balls.pop(self.selected_index)
            self.balls.append(val)
            
            action_details = {'type': 'shift'}
            self.is_animating = True
            self.animation_queue.append({
                'type': 'ball_positions',
                'duration': 15, # frames
                'progress': 0
            })
            return action_details

        # --- Action: Perform Swap (Shift + Left/Right) ---
        elif shift_held and (movement == 3 or movement == 4):
            swap_with_idx = -1
            if movement == 3 and self.selected_index > 0: # Left
                swap_with_idx = self.selected_index - 1
            elif movement == 4 and self.selected_index < self.NUM_BALLS - 1: # Right
                swap_with_idx = self.selected_index + 1
            
            if swap_with_idx != -1:
                # SFX: Swap sound
                self._start_turn()
                
                i, j = self.selected_index, swap_with_idx
                self.balls[i], self.balls[j] = self.balls[j], self.balls[i]
                
                action_details = {'type': 'swap', 'indices': (i, j)}
                self.is_animating = True
                self.animation_queue.append({
                    'type': 'swap',
                    'indices': (i, j),
                    'duration': 12,
                    'progress': 0
                })
                return action_details

        # --- Action: Move Selector (Left/Right) ---
        elif not shift_held and (movement == 3 or movement == 4):
            if movement == 3: # Left
                self.selected_index = max(0, self.selected_index - 1)
            elif movement == 4: # Right
                self.selected_index = min(self.NUM_BALLS - 1, self.selected_index + 1)
            # SFX: UI tick
            action_details = {'type': 'select'}
            return action_details
        
        return None # No meaningful action taken

    def _start_turn(self):
        """Marks the beginning of a turn-consuming action."""
        self.turn_count += 1

    def _end_turn(self):
        """Called when an action animation finishes. Finalizes the turn."""
        turn_reward = 0
        
        # Continuous reward for sorted balls
        correctly_placed = 0
        for i in range(self.NUM_BALLS):
            if self.balls[i] == i + 1:
                correctly_placed += 1
        turn_reward += correctly_placed

        # Check for win condition
        is_sorted = correctly_placed == self.NUM_BALLS
        if is_sorted:
            self.win_state = "WIN"
            turn_reward += 100
            self.game_over = True
            # SFX: Victory fanfare
            for i in range(self.NUM_BALLS):
                self._spawn_particles(self._get_ball_target_pos(i), self.BALL_COLORS[i], 20, 5)

        # Check for loss condition
        elif self.turn_count >= self.MAX_TURNS:
            self.win_state = "LOSE"
            turn_reward -= 100
            self.game_over = True
            # SFX: Failure buzzer

        self.score += turn_reward

    def _update_animations(self):
        # --- Update Ball Positions ---
        for i in range(self.NUM_BALLS):
            target_pos = self._get_ball_target_pos(i)
            self.ball_visual_pos[i] = self.ball_visual_pos[i].lerp(target_pos, 0.2)
        
        # --- Update Selector Position ---
        selector_target = self._get_ball_target_pos(self.selected_index)
        self.selector_visual_pos = self.selector_visual_pos.lerp(selector_target, 0.3)

        # --- Process Animation Queue ---
        if self.animation_queue:
            anim = self.animation_queue[0]
            anim['progress'] += 1
            
            if anim['type'] == 'swap':
                i, j = anim['indices']
                pos_i = self._get_ball_target_pos(i)
                pos_j = self._get_ball_target_pos(j)
                midpoint = pos_i.lerp(pos_j, 0.5) + pygame.Vector2(0, -30)
                
                t = anim['progress'] / anim['duration']
                self.ball_visual_pos[i] = pos_j.lerp(midpoint, 1 - (2*t-1)**2)
                self.ball_visual_pos[j] = pos_i.lerp(midpoint, 1 - (2*t-1)**2)

            if anim['progress'] >= anim['duration']:
                self.animation_queue.pop(0)
                if not self.animation_queue:
                    self.is_animating = False
                    self._end_turn()

        # --- Update Particles ---
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Conveyor Belt ---
        conveyor_rect = pygame.Rect(30, 150, self.SCREEN_WIDTH - 60, 100)
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR, conveyor_rect, border_radius=10)
        pygame.draw.rect(self.screen, self.COLOR_CONVEYOR_BORDER, conveyor_rect, width=3, border_radius=10)

        # --- Draw Target Slots ---
        for i in range(self.NUM_BALLS):
            pos = self._get_ball_target_pos(i)
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 22, self.COLOR_CONVEYOR_BORDER)

        # --- Draw Selector ---
        if not self.game_over:
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2 # 0 to 1
            radius = 28 + pulse * 4
            alpha = 100 + pulse * 100
            
            # Create a temporary surface for the glowing circle
            glow_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(glow_surf, int(radius), int(radius), int(radius-1), (*self.COLOR_SELECTOR, int(alpha)))
            pygame.gfxdraw.filled_circle(glow_surf, int(radius), int(radius), int(radius-1), (*self.COLOR_SELECTOR, int(alpha/4)))
            self.screen.blit(glow_surf, (self.selector_visual_pos.x - radius, self.selector_visual_pos.y - radius))

        # --- Draw Balls ---
        for i in range(self.NUM_BALLS):
            num = self.balls[i]
            pos = self.ball_visual_pos[i]
            color = self.BALL_COLORS[num - 1]
            
            # Draw ball shadow
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y) + 3, 20, (0,0,0,50))
            # Draw ball
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 20, color)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 20, color)
            
            # Draw number on ball
            num_text = self.font_ball.render(str(num), True, self.COLOR_BG)
            text_rect = num_text.get_rect(center=(int(pos.x), int(pos.y)))
            self.screen.blit(num_text, text_rect)
            
        # --- Draw Particles ---
        for p in self.particles:
            size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
            pygame.draw.rect(self.screen, p['color'], (p['pos'].x, p['pos'].y, size, size))

    def _render_ui(self):
        # --- Draw Turn Counter ---
        turn_text = self.font_main.render(f"Turn: {self.turn_count} / {self.MAX_TURNS}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (20, 20))

        # --- Draw Score ---
        score_text = self.font_main.render(f"Score: {self.score}", True, self.COLOR_SCORE)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        # --- Draw Win/Loss Message ---
        if self.win_state:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win_state == "WIN" else "GAME OVER"
            color = self.COLOR_WIN if self.win_state == "WIN" else self.COLOR_LOSE
            end_text = self.font_endgame.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

    def _get_ball_target_pos(self, index):
        spacing = (self.SCREEN_WIDTH - 120) / (self.NUM_BALLS - 1)
        x = 60 + index * spacing
        y = 200
        return pygame.Vector2(x, y)

    def _spawn_particles(self, pos, color, count, speed_mult=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3) * speed_mult
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': self.np_random.integers(20, 41),
                'max_life': 40,
                'color': color,
                'size': self.np_random.integers(2, 5)
            })

    def _get_info(self):
        is_sorted = all(self.balls[i] == i + 1 for i in range(self.NUM_BALLS))
        return {
            "score": self.score,
            "steps": self.steps,
            "turn": self.turn_count,
            "is_sorted": is_sorted,
        }

    def close(self):
        pygame.quit()


# --- Example Usage ---
if __name__ == '__main__':
    # Un-comment the line below to run with a display window
    os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Manual Play Setup ---
    pygame.display.set_caption("Conveyor Sort")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print("Left/Right Arrow: Move selector")
    print("Shift + Left/Right: Swap ball")
    print("Spacebar: Shift ball to end")
    print("R: Reset environment")
    print("Q: Quit")

    last_action_time = pygame.time.get_ticks()
    action_cooldown = 150 # ms

    while running:
        action = [0, 0, 0] # Default no-op
        
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"--- Env Reset ---")
                    continue

        if current_time - last_action_time > action_cooldown:
            keys = pygame.key.get_pressed()
            movement, space, shift = 0, 0, 0
            
            if keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4
            
            if keys[pygame.K_SPACE]:
                space = 1
            
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
            
            action = [movement, space, shift]
            
            # Only step if a meaningful action is taken
            if any(action):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                last_action_time = current_time

                if reward != 0:
                    print(f"Step: {info['steps']}, Turn: {info['turn']}, Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")

                if terminated:
                    print(f"--- Game Over ---")
                    print(f"Final Score: {info['score']}, Final Turn: {info['turn']}")
            else:
                # Since auto_advance is false, we just re-render the current state
                obs = env._get_observation()

        # Draw the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the speed for manual play
        clock.tick(60)

    env.close()