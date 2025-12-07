
# Generated: 2025-08-27T17:22:38.136420
# Source Brief: brief_01510.md
# Brief Index: 1510

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all pixels simultaneously in that direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Push colored pixels to their matching goal zones in this fast-paced arcade puzzle. "
        "Clear the board before the timer runs out!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.GRID_WIDTH, self.GRID_HEIGHT = 16, 10
        self.CELL_SIZE = 40
        self.NUM_PIXELS = 15
        self.MAX_TIME_SECONDS = 60
        self.MOVE_ANIMATION_FRAMES = 8 # How many frames a push animation takes
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 45, 65)
        self.PIXEL_COLORS = [
            (255, 0, 77), (0, 228, 54), (41, 173, 255), (255, 204, 0),
            (255, 117, 56), (199, 0, 255), (0, 255, 188), (255, 89, 201),
            (240, 240, 240), (255, 163, 0), (99, 255, 133), (128, 128, 255),
            (255, 128, 128), (128, 255, 128), (249, 217, 123)
        ]

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state variables (initialized in reset)
        self.pixels = []
        self.goals = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.timer_frames = 0
        self.move_in_progress = 0
        self.game_over = False
        self.rng = None

        # Initialize state
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            # Fallback if seed is not provided, though gym usually provides one
            if self.rng is None:
                self.rng = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.timer_frames = self.MAX_TIME_SECONDS * self.FPS
        self.game_over = False
        self.move_in_progress = 0
        self.particles.clear()
        
        self._generate_level()
        
        return self._get_observation(), self._get_info()

    def _generate_level(self):
        self.pixels.clear()
        self.goals.clear()
        
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.rng.shuffle(all_cells)
        
        chosen_colors = self.rng.choice(self.PIXEL_COLORS, self.NUM_PIXELS, replace=False)

        for i in range(self.NUM_PIXELS):
            goal_pos = all_cells.pop()
            pixel_pos = all_cells.pop()
            color = tuple(chosen_colors[i])

            self.goals.append({
                "pos": goal_pos,
                "color": color
            })
            
            self.pixels.append({
                "id": i,
                "grid_pos": pixel_pos,
                "draw_pos": (pixel_pos[0] * self.CELL_SIZE, pixel_pos[1] * self.CELL_SIZE),
                "color": color,
                "goal_pos": goal_pos,
                "in_goal": False
            })

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        self.steps += 1
        self.timer_frames -= 1

        # Handle player input if no animation is playing
        if self.move_in_progress == 0 and movement > 0:
            # SFX: PUSH
            reward = self._handle_push(movement)
            self.move_in_progress = self.MOVE_ANIMATION_FRAMES
        
        if self.move_in_progress > 0:
            self.move_in_progress -= 1
            
        # Update animations every frame
        self._update_animations()

        terminated, terminal_reward = self._check_termination()
        if terminated:
            reward += terminal_reward
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_push(self, movement):
        # 0=none, 1=up, 2=down, 3=left, 4=right
        move_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        dx, dy = move_map[movement]

        # Sort pixels to handle collisions correctly
        # For a right push (dx=1), sort by x descending. For a left push (dx=-1), sort by x ascending.
        # For a down push (dy=1), sort by y descending. For an up push (dy=-1), sort by y ascending.
        sort_key = 0 if dx != 0 else 1
        sort_reverse = dx > 0 or dy > 0
        sorted_pixels = sorted(self.pixels, key=lambda p: p["grid_pos"][sort_key], reverse=sort_reverse)

        occupied_cells = {p["grid_pos"] for p in self.pixels}
        new_positions = {}
        move_reward = 0

        for p in sorted_pixels:
            if p["in_goal"]:
                new_positions[p["id"]] = p["grid_pos"]
                continue

            old_dist = abs(p["grid_pos"][0] - p["goal_pos"][0]) + abs(p["grid_pos"][1] - p["goal_pos"][1])
            
            target_pos = (p["grid_pos"][0] + dx, p["grid_pos"][1] + dy)
            
            # Check boundaries
            if not (0 <= target_pos[0] < self.GRID_WIDTH and 0 <= target_pos[1] < self.GRID_HEIGHT):
                new_positions[p["id"]] = p["grid_pos"]
                continue

            # Check collisions with other pixels (use their final intended positions)
            is_blocked = False
            for other_p in self.pixels:
                if other_p["id"] == p["id"]:
                    continue
                # If the target is occupied by a pixel that has already been processed (and will move)
                # or is not yet processed (and will stay or move away)
                if new_positions.get(other_p["id"], other_p["grid_pos"]) == target_pos:
                    is_blocked = True
                    break
            
            if is_blocked:
                new_positions[p["id"]] = p["grid_pos"]
            else:
                new_positions[p["id"]] = target_pos
                
                new_dist = abs(target_pos[0] - p["goal_pos"][0]) + abs(target_pos[1] - p["goal_pos"][1])
                if new_dist < old_dist:
                    move_reward += 0.1
                elif new_dist > old_dist:
                    move_reward -= 0.2
        
        # Apply new positions and check for goal completion
        for p in self.pixels:
            p["grid_pos"] = new_positions[p["id"]]
            if not p["in_goal"] and p["grid_pos"] == p["goal_pos"]:
                p["in_goal"] = True
                self.score += 10
                move_reward += 1
                self._create_particles(p["grid_pos"], p["color"])
                # SFX: GOAL_MET

        return move_reward

    def _check_termination(self):
        # Win condition: all pixels in their goals
        if all(p["in_goal"] for p in self.pixels):
            # SFX: WIN
            return True, 50
        
        # Loss condition: timer runs out
        if self.timer_frames <= 0:
            # SFX: LOSE
            return True, -100
        
        return False, 0

    def _update_animations(self):
        # Smoothly move pixels to their target grid positions
        for p in self.pixels:
            target_draw_x = p["grid_pos"][0] * self.CELL_SIZE
            target_draw_y = p["grid_pos"][1] * self.CELL_SIZE
            
            current_x, current_y = p["draw_pos"]
            # Lerp for smooth movement
            p["draw_pos"] = (
                current_x + (target_draw_x - current_x) * 0.3,
                current_y + (target_draw_y - current_y) * 0.3
            )
            
        # Update particles
        for particle in self.particles[:]:
            particle["pos"][0] += particle["vel"][0]
            particle["pos"][1] += particle["vel"][1]
            particle["vel"][1] += 0.1 # Gravity
            particle["life"] -= 1
            if particle["life"] <= 0:
                self.particles.remove(particle)

    def _create_particles(self, grid_pos, color):
        center_x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(30):
            angle = self.rng.uniform(0, 2 * math.pi)
            speed = self.rng.uniform(1, 4)
            self.particles.append({
                "pos": [center_x, center_y],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": self.rng.integers(15, 30),
                "color": color,
                "size": self.rng.uniform(2, 5)
            })

    def _get_info(self):
        pixels_in_goal = sum(1 for p in self.pixels if p["in_goal"])
        return {
            "score": self.score,
            "steps": self.steps,
            "timer_seconds": max(0, self.timer_frames // self.FPS),
            "pixels_in_goal": pixels_in_goal,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
            
        # Draw goals
        for goal in self.goals:
            r, g, b = goal["color"]
            goal_color = (max(0, r-80), max(0, g-80), max(0, b-80))
            rect = pygame.Rect(goal["pos"][0] * self.CELL_SIZE, goal["pos"][1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, goal_color, rect)
            pygame.draw.rect(self.screen, goal["color"], rect.inflate(-8, -8), border_radius=4)

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p["life"] / 30.0))))
            color = p["color"] + (alpha,)
            size = p["size"] * (p["life"] / 30.0)
            if size > 1:
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p["pos"][0]), int(p["pos"][1]), int(size), color
                )

        # Draw pixels
        for p in self.pixels:
            padding = 2
            rect = pygame.Rect(
                p["draw_pos"][0] + padding, p["draw_pos"][1] + padding,
                self.CELL_SIZE - padding*2, self.CELL_SIZE - padding*2
            )
            shadow_rect = rect.move(2, 2)
            pygame.draw.rect(self.screen, (0,0,0,100), shadow_rect, border_radius=6)
            pygame.draw.rect(self.screen, p["color"], rect, border_radius=6)
            
            # Add a white highlight
            highlight_rect = rect.inflate(-16, -16)
            highlight_rect.topleft = (rect.left + 4, rect.top + 4)
            pygame.draw.rect(self.screen, (255,255,255,80), highlight_rect, border_radius=3)
            
    def _render_ui(self):
        info = self._get_info()
        
        # Score display
        score_text = self.font_medium.render(f"SCORE: {info['score']}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 10))
        
        # Timer display
        timer_val = info['timer_seconds']
        timer_color = (255, 255, 255) if timer_val > 10 else (255, 100, 100)
        timer_text = self.font_medium.render(f"TIME: {timer_val}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)
        
        # Pixels in goal display
        goal_count_text = self.font_small.render(f"{info['pixels_in_goal']} / {self.NUM_PIXELS} IN GOAL", True, (200, 200, 220))
        goal_count_rect = goal_count_text.get_rect(midbottom=(self.WIDTH / 2, self.HEIGHT - 10))
        self.screen.blit(goal_count_text, goal_count_rect)

        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = all(p["in_goal"] for p in self.pixels)
            msg = "LEVEL CLEAR!" if win_condition else "TIME UP!"
            color = (100, 255, 150) if win_condition else (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            end_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, end_rect)

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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Pixel Pusher")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()