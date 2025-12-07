import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move your character (yellow square). "
        "Push colored blocks to their matching goal outlines. "
        "Blocks slide until they hit an obstacle."
    )

    game_description = (
        "A retro-style puzzle game where you push blocks to their goals. "
        "Plan your moves carefully, as you have a limited number of pushes "
        "to solve each stage."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_size = (640, 400)
        self.screen = pygame.Surface(self.screen_size)
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 28)
            self.font_msg = pygame.font.SysFont(None, 52)

        # --- Visuals ---
        self.COLOR_BG = (15, 15, 25)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_PUSHER = (255, 255, 0)
        self.BLOCK_COLORS = [
            (255, 50, 50),   # Red
            (50, 150, 255),  # Blue
            (50, 255, 150),  # Green
            (255, 150, 50),  # Orange
            (200, 50, 255),  # Purple
            (50, 255, 255),  # Cyan
        ]
        self.GOAL_COLORS = [pygame.Color(c).lerp(self.COLOR_BG, 0.5) for c in self.BLOCK_COLORS]
        self.PARTICLE_COLOR = (200, 200, 220)

        # --- Game Config ---
        self.max_steps = 1000
        self.stage_configs = self._define_stages()
        self.current_stage_index = 0

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        
        self.grid_size = (0, 0)
        self.tile_size = 0
        self.grid_offset = pygame.Vector2(0, 0)
        
        self.pusher_pos = pygame.Vector2(0, 0)
        self.blocks = []
        self.moves_left = 0
        self.particles = []

        # self.reset() is called by the test runner, no need to call it here.
        
    def _define_stages(self):
        return [
            { # Stage 1: Simple
                "grid_size": (8, 6), "moves": 40,
                "pusher_start": (1, 3),
                "blocks": [
                    {"pos": (3, 2), "color_idx": 0},
                    {"pos": (4, 4), "color_idx": 1},
                ],
                "goals": [
                    {"pos": (6, 2), "color_idx": 0},
                    {"pos": (6, 4), "color_idx": 1},
                ],
            },
            { # Stage 2: Requires a bit of planning
                "grid_size": (10, 8), "moves": 80,
                "pusher_start": (1, 4),
                "blocks": [
                    {"pos": (3, 2), "color_idx": 0},
                    {"pos": (3, 6), "color_idx": 1},
                    {"pos": (7, 4), "color_idx": 2},
                ],
                "goals": [
                    {"pos": (8, 2), "color_idx": 0},
                    {"pos": (8, 6), "color_idx": 1},
                    {"pos": (1, 1), "color_idx": 2},
                ],
            },
            { # Stage 3: More complex interactions
                "grid_size": (12, 10), "moves": 120,
                "pusher_start": (1, 5),
                "blocks": [
                    {"pos": (3, 3), "color_idx": 0},
                    {"pos": (3, 7), "color_idx": 1},
                    {"pos": (8, 2), "color_idx": 2},
                    {"pos": (8, 8), "color_idx": 3},
                ],
                "goals": [
                    {"pos": (10, 5), "color_idx": 1},
                    {"pos": (1, 1), "color_idx": 2},
                    {"pos": (10, 1), "color_idx": 3},
                    {"pos": (5, 5), "color_idx": 0},
                ],
            },
        ]
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.win_message = ""
        self.particles.clear()
        
        # Cycle through stages on each reset
        config = self.stage_configs[self.current_stage_index]
        self.current_stage_index = (self.current_stage_index + 1) % len(self.stage_configs)

        self.grid_size = config["grid_size"]
        self.moves_left = config["moves"]
        self.pusher_pos = pygame.Vector2(config["pusher_start"])

        # Calculate grid rendering properties
        grid_w_pixels = self.screen_size[0] * 0.8
        grid_h_pixels = self.screen_size[1] * 0.8
        self.tile_size = int(min(grid_w_pixels / self.grid_size[0], grid_h_pixels / self.grid_size[1]))
        self.grid_offset.x = (self.screen_size[0] - self.grid_size[0] * self.tile_size) / 2
        self.grid_offset.y = (self.screen_size[1] - self.grid_size[1] * self.tile_size) / 2

        self.blocks = []
        goals_map = {g['color_idx']: pygame.Vector2(g['pos']) for g in config['goals']}
        for b in config['blocks']:
            self.blocks.append({
                "pos": pygame.Vector2(b['pos']),
                "color_idx": b['color_idx'],
                "color": self.BLOCK_COLORS[b['color_idx']],
                "goal_pos": goals_map[b['color_idx']],
                "on_goal": False,
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement = action[0]
        self.steps += 1
        reward = 0.0
        
        # --- Action Logic ---
        if movement != 0:
            self.moves_left -= 1
            # SFX: Step
            
            direction = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            dx, dy = direction
            
            target_pos = self.pusher_pos + pygame.Vector2(dx, dy)
            
            # Check if target is a block
            pushed_block = None
            for block in self.blocks:
                if block['pos'] == target_pos:
                    pushed_block = block
                    break
            
            if pushed_block:
                reward += self._handle_push(pushed_block, dx, dy)
            else:
                reward += self._handle_move(target_pos)
        
        # --- Update & Check State ---
        self._update_particles()
        self.score += reward
        
        terminated = self._check_termination()
        if terminated:
            if self.win_message == "PUZZLE SOLVED!":
                reward += 100.0
                # SFX: Win Jingle
            else:
                reward -= 100.0
                # SFX: Lose Fanfare
            self.score += reward

        truncated = self.steps >= self.max_steps
        if truncated:
            self.game_over = True
            self.win_message = "TIME LIMIT REACHED"

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_push(self, block, dx, dy):
        reward = 0.0
        
        if block['on_goal']:
            # SFX: Blocked/Locked
            reward -= 0.01 # Penalty for pushing a locked block
            return reward

        old_dist = self._manhattan_distance(block['pos'], block['goal_pos'])
        old_pos = pygame.Vector2(block['pos']) # FIX: pygame.Vector2 does not have .copy()

        # Find slide destination
        slide_pos = pygame.Vector2(block['pos']) # FIX: pygame.Vector2 does not have .copy()
        while True:
            next_pos = slide_pos + pygame.Vector2(dx, dy)
            if not (0 <= next_pos.x < self.grid_size[0] and 0 <= next_pos.y < self.grid_size[1]):
                break # Wall collision
            
            if any(b['pos'] == next_pos for b in self.blocks if b is not block):
                break # Other block collision
                
            slide_pos = next_pos
        
        if slide_pos != old_pos:
            # SFX: Block Slide
            self.pusher_pos = old_pos
            block['pos'] = slide_pos
            self._create_slide_particles(old_pos, slide_pos, dx, dy)

            new_dist = self._manhattan_distance(block['pos'], block['goal_pos'])

            if new_dist < old_dist:
                reward += 0.1
            elif new_dist > old_dist:
                reward -= 0.01

            if block['pos'] == block['goal_pos']:
                block['on_goal'] = True
                reward += 1.0
                # SFX: Goal Lock
        else:
            # SFX: Blocked/Thud
            reward -= 0.05 # Penalty for a failed push
            
        return reward

    def _handle_move(self, target_pos):
        if 0 <= target_pos.x < self.grid_size[0] and 0 <= target_pos.y < self.grid_size[1]:
            self.pusher_pos = target_pos
            return -0.001 # Small cost for moving
        return 0.0 # No change if moving into a wall

    def _manhattan_distance(self, v1, v2):
        return abs(v1.x - v2.x) + abs(v1.y - v2.y)

    def _check_termination(self):
        if all(b['on_goal'] for b in self.blocks):
            self.game_over = True
            self.win_message = "PUZZLE SOLVED!"
            return True
        
        if self.moves_left <= 0:
            self.game_over = True
            self.win_message = "OUT OF MOVES"
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}
        
    def _world_to_screen(self, pos):
        x = self.grid_offset.x + pos.x * self.tile_size
        y = self.grid_offset.y + pos.y * self.tile_size
        return pygame.Vector2(x, y)

    def _render_game(self):
        # Draw grid
        for i in range(self.grid_size[0] + 1):
            x = self.grid_offset.x + i * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.grid_offset.y), (x, self.grid_offset.y + self.grid_size[1] * self.tile_size))
        for i in range(self.grid_size[1] + 1):
            y = self.grid_offset.y + i * self.tile_size
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.grid_offset.x, y), (self.grid_offset.x + self.grid_size[0] * self.tile_size, y))

        # Draw goals
        for block in self.blocks:
            if not block['on_goal']:
                screen_pos = self._world_to_screen(block['goal_pos'])
                rect = pygame.Rect(screen_pos.x, screen_pos.y, self.tile_size, self.tile_size)
                goal_color = self.GOAL_COLORS[block['color_idx']]
                pygame.gfxdraw.rectangle(self.screen, rect, goal_color)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            try:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), color_with_alpha)
            except TypeError: # Older pygame might not like tuple for color
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['size']), p['color'])


        # Draw blocks
        for block in self.blocks:
            screen_pos = self._world_to_screen(block['pos'])
            rect = pygame.Rect(screen_pos.x, screen_pos.y, self.tile_size, self.tile_size)
            
            if block['on_goal']:
                # Draw a bright, filled-in version
                pygame.draw.rect(self.screen, block['color'], rect)
                inner_rect = rect.inflate(-self.tile_size * 0.4, -self.tile_size * 0.4)
                pygame.draw.rect(self.screen, (255, 255, 255), inner_rect, border_radius=2)
            else:
                # Draw a standard block
                pygame.draw.rect(self.screen, block['color'], rect.inflate(-self.tile_size * 0.1, -self.tile_size * 0.1), border_radius=3)
        
        # Draw pusher
        pusher_rect = pygame.Rect(0, 0, self.tile_size, self.tile_size)
        pusher_rect.center = self._world_to_screen(self.pusher_pos) + pygame.Vector2(self.tile_size/2, self.tile_size/2)
        pygame.draw.rect(self.screen, self.COLOR_PUSHER, pusher_rect.inflate(-self.tile_size * 0.3, -self.tile_size * 0.3), border_radius=3)

    def _render_ui(self):
        # Render moves and score
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, (200, 200, 255))
        score_text = self.font_ui.render(f"Score: {self.score:.2f}", True, (200, 200, 255))
        self.screen.blit(moves_text, (10, 10))
        self.screen.blit(score_text, (10, 40))
        
        # Render game over message
        if self.game_over:
            overlay = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_color = (100, 255, 100) if "SOLVED" in self.win_message else (255, 100, 100)
            msg_surf = self.font_msg.render(self.win_message, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.screen_size[0] / 2, self.screen_size[1] / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _create_slide_particles(self, start_pos_grid, end_pos_grid, dx, dy):
        dist = int(self._manhattan_distance(start_pos_grid, end_pos_grid))
        if dist == 0: return
        for i in range(dist * 3):
            progress = i / (dist * 3)
            pos_grid = start_pos_grid.lerp(end_pos_grid, progress)
            screen_pos = self._world_to_screen(pos_grid) + pygame.Vector2(self.tile_size/2, self.tile_size/2)
            
            self.particles.append({
                'pos': screen_pos + pygame.Vector2(random.uniform(-5, 5), random.uniform(-5, 5)),
                'vel': pygame.Vector2(dx, dy) * random.uniform(-0.5, 0.5),
                'life': random.randint(15, 30),
                'max_life': 30,
                'size': random.uniform(1, 4),
                'color': self.PARTICLE_COLOR
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['vel'] *= 0.9

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game directly
    # To run, you might need to comment out the SDL_VIDEODRIVER line at the top
    # and install pygame: pip install pygame
    
    # Re-enable display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Pusher Puzzle")
    screen = pygame.display.set_mode(env.screen_size)
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print(f"Game: {env.game_description}")
    print(f"Controls: {env.user_guide}")
    print("Press 'R' to reset the environment.")
    print("="*30 + "\n")

    last_action_time = 0
    action_delay = 150 # ms

    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Environment Reset. New Stage Loaded. Info: {info}")

        # Map keys to actions with a delay to prevent rapid-fire moves
        current_time = pygame.time.get_ticks()
        if current_time - last_action_time > action_delay:
            keys = pygame.key.get_pressed()
            moved = False
            if keys[pygame.K_UP]:
                action[0] = 1
                moved = True
            elif keys[pygame.K_DOWN]:
                action[0] = 2
                moved = True
            elif keys[pygame.K_LEFT]:
                action[0] = 3
                moved = True
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
                moved = True
            
            if keys[pygame.K_SPACE]:
                action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                action[2] = 1

            if moved:
                last_action_time = current_time
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}, Moves Left: {info['moves_left']}")
                
                if terminated or truncated:
                    print(f"Episode Finished. Final Score: {info['score']:.2f}")
                    # Render final state before pausing
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    
                    # Wait for a moment before auto-resetting
                    pygame.time.wait(2000)
                    obs, info = env.reset()
                    total_reward = 0
                    print(f"Environment Reset. New Stage Loaded. Info: {info}")

        # Render the observation to the display window
        current_obs, _ = env._get_observation(), env._get_info()
        surf = pygame.surfarray.make_surface(np.transpose(current_obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    env.close()