
# Generated: 2025-08-28T05:39:43.310993
# Source Brief: brief_05652.md
# Brief Index: 5652

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to push all slidable boxes left or right."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A physics-based puzzle game. Push all the boxes onto the platforms within 20 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.MOVE_LIMIT = 20
        self.NUM_BOXES = 5
        self.BOX_SIZE = 40
        self.GRAVITY = 4
        self.PUSH_FORCE = 8

        # Colors
        self.COLOR_BG_TOP = (135, 206, 235)  # Sky Blue
        self.COLOR_BG_BOTTOM = (176, 224, 230)  # Powder Blue
        self.COLOR_BOX = (220, 20, 60)  # Crimson
        self.COLOR_BOX_SHADOW = (139, 0, 0)  # Dark Red
        self.COLOR_PLATFORM = (169, 169, 169)  # Dark Gray
        self.COLOR_PLATFORM_SHADOW = (105, 105, 105) # Dim Gray
        self.COLOR_PLAYER_EFFECT = (255, 215, 0, 150) # Gold, semi-transparent
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_SHADOW = (0, 0, 0)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 36)
        self.font_msg = pygame.font.Font(None, 72)
        
        # Initialize state variables
        self.box_rects = []
        self.platform_rects = []
        self.placed_box_ids = set()
        self.moves_made = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.push_effect = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_made = 0
        self.score = 0
        self.game_over = False
        self.win_message = ""
        self.placed_box_ids = set()
        self.push_effect = None

        # Define platforms (hardcoded for a consistent puzzle)
        self.platform_rects = [
            pygame.Rect(100, self.SCREEN_HEIGHT - 50, 120, 15),
            pygame.Rect(self.SCREEN_WIDTH - 220, self.SCREEN_HEIGHT - 50, 120, 15),
            pygame.Rect(260, self.SCREEN_HEIGHT - 150, 120, 15),
            pygame.Rect(180, self.SCREEN_HEIGHT - 250, 80, 15),
            pygame.Rect(self.SCREEN_WIDTH - 260, self.SCREEN_HEIGHT - 250, 80, 15),
        ]

        # Define initial box positions (randomized on the floor)
        self.box_rects = []
        spawn_y = self.SCREEN_HEIGHT - self.BOX_SIZE
        for i in range(self.NUM_BOXES):
            while True:
                spawn_x = self.np_random.integers(0, self.SCREEN_WIDTH - self.BOX_SIZE)
                new_box = pygame.Rect(spawn_x, spawn_y, self.BOX_SIZE, self.BOX_SIZE)
                if new_box.collidelist(self.box_rects) == -1:
                    self.box_rects.append(new_box)
                    break
        
        # Settle physics initially
        self._apply_physics()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        self.push_effect = None

        is_push_action = movement in [3, 4]

        if is_push_action:
            self.moves_made += 1
            reward -= 0.1 # Cost of making a move

            direction = -1 if movement == 3 else 1 # 3=left, 4=right
            self._apply_horizontal_push(direction)
            self._apply_physics()
            
            # Add a visual effect for the push
            self.push_effect = {"dir": direction, "alpha": 255}

            # Check for newly placed boxes
            current_placed_ids = self._get_placed_box_ids()
            newly_placed_ids = current_placed_ids - self.placed_box_ids
            if newly_placed_ids:
                reward += 1.0 * len(newly_placed_ids)
                # Sound effect placeholder: # sfx_box_placed
            self.placed_box_ids = current_placed_ids
        
        self.score += reward
        terminated = self._check_termination()
        
        if terminated:
            if len(self.placed_box_ids) == self.NUM_BOXES:
                self.score += 10
                reward += 10
                self.win_message = "YOU WIN!"
                # Sound effect placeholder: # sfx_win
            else: # Out of moves
                self.score -= 10
                reward -= 10
                self.win_message = "OUT OF MOVES"
                # Sound effect placeholder: # sfx_lose

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _apply_horizontal_push(self, direction):
        # Use a sort order to process boxes correctly, preventing them from leapfrogging
        sorted_indices = sorted(range(len(self.box_rects)), key=lambda k: self.box_rects[k].x, reverse=(direction == 1))
        
        for _ in range(self.SCREEN_WIDTH // self.PUSH_FORCE): # Iterate enough times to slide across the screen
            moved_this_pass = False
            for i in sorted_indices:
                box = self.box_rects[i]
                original_x = box.x
                box.x += direction * self.PUSH_FORCE

                # Wall collision
                if box.left < 0: box.left = 0
                if box.right > self.SCREEN_WIDTH: box.right = self.SCREEN_WIDTH

                # Box-to-box collision
                collision_list = box.collidelistall(self.box_rects)
                for j in collision_list:
                    if i == j: continue
                    other_box = self.box_rects[j]
                    if direction == 1: # Pushing right
                        box.right = other_box.left
                    else: # Pushing left
                        box.left = other_box.right
                
                if box.x != original_x:
                    moved_this_pass = True
            
            if not moved_this_pass:
                break
        # Sound effect placeholder: # sfx_push_slide

    def _apply_physics(self):
        # Settle boxes using gravity until stable
        for _ in range(self.SCREEN_HEIGHT // self.GRAVITY):
            moved_this_pass = False
            for i, box in enumerate(self.box_rects):
                box.y += self.GRAVITY
                
                # Floor collision
                if box.bottom > self.SCREEN_HEIGHT:
                    box.bottom = self.SCREEN_HEIGHT

                # Collision with other boxes
                collision_list = box.collidelistall(self.box_rects)
                for j in collision_list:
                    if i == j: continue
                    other_box = self.box_rects[j]
                    # If box fell into another, move it back on top
                    if box.bottom > other_box.top and box.top < other_box.top:
                        box.bottom = other_box.top
                
                # Collision with platforms
                for plat in self.platform_rects:
                    if box.colliderect(plat) and box.bottom > plat.top and box.top < plat.top:
                        box.bottom = plat.top
                
                # Using a small tolerance to check for movement
                if abs(box.y - (box.y - self.GRAVITY)) > 0:
                     moved_this_pass = True

            if not moved_this_pass:
                break
    
    def _get_placed_box_ids(self):
        placed_ids = set()
        for i, box in enumerate(self.box_rects):
            # Check if box is on the floor
            if box.bottom == self.SCREEN_HEIGHT:
                continue
            
            # Check if supported by a platform
            is_on_platform = False
            for plat in self.platform_rects:
                # Box is on top of platform if their sides align and box bottom is at platform top
                if box.bottom == plat.top and box.right > plat.left and box.left < plat.right:
                    is_on_platform = True
                    break
            if is_on_platform:
                placed_ids.add(i)
        return placed_ids

    def _check_termination(self):
        win = len(self.placed_box_ids) == self.NUM_BOXES
        lose = self.moves_made >= self.MOVE_LIMIT
        if win or lose:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        # Draw gradient background
        self.screen.fill(self.COLOR_BG_TOP)
        for y in range(self.SCREEN_HEIGHT // 2, self.SCREEN_HEIGHT):
            alpha = int(255 * ((y - self.SCREEN_HEIGHT // 2) / (self.SCREEN_HEIGHT // 2)))
            color = self.COLOR_BG_BOTTOM
            s = pygame.Surface((self.SCREEN_WIDTH, 1))
            s.set_alpha(alpha)
            s.fill(color)
            self.screen.blit(s, (0, y))

        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        shadow_offset = 4
        # Render platforms
        for r in self.platform_rects:
            shadow_rect = r.move(shadow_offset, shadow_offset)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM_SHADOW, shadow_rect)
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, r)

        # Render boxes
        for r in self.box_rects:
            shadow_rect = r.move(shadow_offset, shadow_offset)
            pygame.draw.rect(self.screen, self.COLOR_BOX_SHADOW, shadow_rect)
            pygame.draw.rect(self.screen, self.COLOR_BOX, r)
        
        # Render push effect
        if self.push_effect and self.push_effect["alpha"] > 0:
            surf = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            color = self.COLOR_PLAYER_EFFECT[:3] + (self.push_effect["alpha"],)
            if self.push_effect["dir"] == 1: # Right
                pygame.gfxdraw.filled_trigon(surf, 0, 0, 0, self.SCREEN_HEIGHT, 100, self.SCREEN_HEIGHT // 2, color)
            else: # Left
                pygame.gfxdraw.filled_trigon(surf, self.SCREEN_WIDTH, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT // 2, color)
            self.screen.blit(surf, (0, 0))
            self.push_effect["alpha"] = max(0, self.push_effect["alpha"] - 25)

    def _render_ui(self):
        def draw_text(text, font, color, pos, shadow=True):
            if shadow:
                text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
                self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, pos)

        # Display moves and score
        moves_text = f"Moves: {self.moves_made}/{self.MOVE_LIMIT}"
        score_text = f"Score: {self.score:.1f}"
        draw_text(moves_text, self.font_ui, self.COLOR_TEXT, (10, 10))
        draw_text(score_text, self.font_ui, self.COLOR_TEXT, (10, 45))

        # Display win/loss message
        if self.game_over and self.win_message:
            text_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            shadow_surf = self.font_msg.render(self.win_message, True, self.COLOR_TEXT_SHADOW)
            shadow_rect = shadow_surf.get_rect(center=(self.SCREEN_WIDTH / 2 + 4, self.SCREEN_HEIGHT / 2 + 4))
            
            self.screen.blit(shadow_surf, shadow_rect)
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_made": self.moves_made,
            "boxes_placed": len(self.placed_box_ids),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Box Stacker")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0]) # Default no-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    action = np.array([0, 0, 0])
                elif event.key == pygame.K_q:
                    running = False

        # Only step if a valid key was pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_made']}, Terminated: {terminated}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()