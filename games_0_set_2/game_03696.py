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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys (↑, ↓, ←, →) to push all blocks in that direction."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro-style puzzle game. Push colored blocks onto their matching goals before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    class Particle:
        def __init__(self, pos, vel, color, lifetime):
            self.pos = pygame.math.Vector2(pos)
            self.vel = pygame.math.Vector2(vel)
            self.color = color
            self.lifetime = lifetime
            self.max_lifetime = lifetime

        def update(self):
            self.pos += self.vel
            self.vel *= 0.95 # friction
            self.lifetime -= 1

        def draw(self, surface):
            if self.lifetime > 0:
                progress = self.lifetime / self.max_lifetime
                size = int(max(1, 4 * progress))
                alpha = int(255 * progress)
                s = pygame.Surface((size, size), pygame.SRCALPHA)
                s.fill((self.color.r, self.color.g, self.color.b, alpha))
                surface.blit(s, (int(self.pos.x - size/2), int(self.pos.y - size/2)))

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.TILE_SIZE = 40
        self.GRID_WIDTH, self.GRID_HEIGHT = self.WIDTH // self.TILE_SIZE, self.HEIGHT // self.TILE_SIZE
        self.MAX_STEPS = 1000

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.Font(None, 50)
        self.font_small = pygame.font.Font(None, 30)

        # Colors
        self.COLOR_BG = pygame.Color("#2c3e50")
        self.COLOR_GRID = pygame.Color("#34495e")
        self.COLOR_TEXT = pygame.Color("#ecf0f1")
        self.BLOCK_COLORS = [
            pygame.Color("#e74c3c"), pygame.Color("#2ecc71"), pygame.Color("#3498db"),
            pygame.Color("#f1c40f"), pygame.Color("#9b59b6"), pygame.Color("#1abc9c"),
            pygame.Color("#e67e22"), pygame.Color("#7f8c8d")
        ]
        self.GOAL_COLOR_MOD = pygame.Color('gray30')

        # Initialize state variables
        # self.reset() is called by the test harness, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.level = 1
        self.game_over = False
        self.level_complete_msg_timer = 0
        self.particles = []
        self.push_effects = []
        
        self._generate_level(self.level)

        return self._get_observation(), self._get_info()

    def _generate_level(self, level_num):
        num_blocks = min(
            self.GRID_WIDTH * self.GRID_HEIGHT // 3,
            5 + (level_num - 1) * 2
        )
        self.moves_left = 120 + (level_num - 1) * 20
        self.max_moves = self.moves_left

        self.blocks = []
        self.goals = []
        self.goal_map = {}

        possible_positions = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        self.np_random.shuffle(possible_positions)

        # Ensure blocks don't start on goals
        goal_positions = possible_positions[:num_blocks]
        block_positions = possible_positions[num_blocks:num_blocks*2]
        
        colors = self.BLOCK_COLORS[:]
        self.np_random.shuffle(colors)

        for i in range(num_blocks):
            color = colors[i % len(colors)]
            goal_pos = goal_positions[i]
            block_pos = block_positions[i]
            
            self.goals.append({'pos': goal_pos, 'color': color, 'id': i})
            self.blocks.append({'pos': block_pos, 'color': color, 'id': i})
            self.goal_map[i] = goal_pos
        
        self.level_complete_msg_timer = 60 # Show "Level X" for a bit

    def step(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        reward = -0.1 # Small penalty for taking a step to encourage efficiency
        terminated = False
        
        self.steps += 1
        if self.level_complete_msg_timer > 0:
            self.level_complete_msg_timer -= 1

        if movement > 0:
            self.moves_left -= 1
            move_reward, _ = self._handle_push(movement)
            reward += move_reward
        
        win = self._check_win_condition()

        if win:
            # sfx: level_complete.wav
            reward += 100
            self.score += self.moves_left # Time bonus
            self.level += 1
            self._generate_level(self.level)
            # The game continues to the next level, so not terminated
        elif self.moves_left <= 0 or self.steps >= self.MAX_STEPS:
            # sfx: game_over.wav
            reward -= 50
            self.game_over = True
            terminated = True
        
        truncated = False
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_push(self, movement):
        # 1=up, 2=down, 3=left, 4=right
        if movement == 1: dx, dy, sort_rev, sort_idx = 0, -1, False, 1
        elif movement == 2: dx, dy, sort_rev, sort_idx = 0, 1, True, 1
        elif movement == 3: dx, dy, sort_rev, sort_idx = -1, 0, False, 0
        elif movement == 4: dx, dy, sort_rev, sort_idx = 1, 0, True, 0
        else: return 0, False

        self.push_effects.append({'dir': movement, 'life': 5})
        
        pre_push_dists = {b['id']: self._manhattan_dist(b['pos'], self.goal_map[b['id']]) for b in self.blocks}
        pre_on_goal = {b['id'] for b in self.blocks if b['pos'] == self.goal_map[b['id']]}

        sorted_blocks = sorted(self.blocks, key=lambda b: b['pos'][sort_idx], reverse=sort_rev)
        
        moved_any = False
        
        # This logic processes blocks in order, allowing them to push chains
        for block_to_process in sorted_blocks:
            # Find the full chain connected to this block in the push direction
            chain = []
            current_pos = block_to_process['pos']
            
            while True:
                found_block = next((b for b in self.blocks if b['pos'] == current_pos), None)
                if found_block:
                    chain.append(found_block)
                    current_pos = (current_pos[0] + dx, current_pos[1] + dy)
                else:
                    break
            
            if not chain: continue

            # Check if the head of the chain is blocked
            head_next_pos = (chain[-1]['pos'][0] + dx, chain[-1]['pos'][1] + dy)
            if not (0 <= head_next_pos[0] < self.GRID_WIDTH and 0 <= head_next_pos[1] < self.GRID_HEIGHT):
                continue # Blocked by wall
            if any(b['pos'] == head_next_pos for b in self.blocks if b not in chain):
                continue # Blocked by another block not in this chain

            # If not blocked, move the entire chain
            for b_in_chain in reversed(chain):
                old_pos = b_in_chain['pos']
                new_pos = (old_pos[0] + dx, old_pos[1] + dy)
                b_in_chain['pos'] = new_pos
                moved_any = True
                
                # sfx: block_slide.wav
                for _ in range(5):
                    px = (old_pos[0] + 0.5) * self.TILE_SIZE
                    py = (old_pos[1] + 0.5) * self.TILE_SIZE
                    vel_x = -dx * self.np_random.uniform(1, 4) + self.np_random.uniform(-1, 1)
                    vel_y = -dy * self.np_random.uniform(1, 4) + self.np_random.uniform(-1, 1)
                    life = self.np_random.integers(10, 20)
                    self.particles.append(self.Particle((px, py), (vel_x, vel_y), b_in_chain['color'], life))

        if not moved_any:
            return 0, False

        # Calculate reward
        reward = 0
        post_push_dists = {b['id']: self._manhattan_dist(b['pos'], self.goal_map[b['id']]) for b in self.blocks}
        post_on_goal = {b['id'] for b in self.blocks if b['pos'] == self.goal_map[b['id']]}

        for b_id in post_push_dists:
            dist_change = pre_push_dists[b_id] - post_push_dists[b_id]
            reward += dist_change # +1 for closer, -1 for further

        newly_on_goal = post_on_goal - pre_on_goal
        for _ in newly_on_goal:
            # sfx: goal_reached.wav
            reward += 10
            
        return reward, True

    def _manhattan_dist(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _check_win_condition(self):
        if not self.blocks: return False
        return all(block['pos'] == self.goal_map[block['id']] for block in self.blocks)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.WIDTH, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.TILE_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

        # Draw goals
        for goal in self.goals:
            x, y = goal['pos']
            rect = pygame.Rect(x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
            goal_color = goal['color'].lerp(self.GOAL_COLOR_MOD, 0.6)
            pygame.draw.rect(self.screen, goal_color, rect)
            pygame.gfxdraw.rectangle(self.screen, rect, goal['color'])

        # Draw blocks
        block_size = self.TILE_SIZE - 6
        offset = (self.TILE_SIZE - block_size) // 2
        for block in self.blocks:
            x, y = block['pos']
            rect = pygame.Rect(x * self.TILE_SIZE + offset, y * self.TILE_SIZE + offset, block_size, block_size)
            pygame.draw.rect(self.screen, block['color'], rect, border_radius=4)
            pygame.draw.rect(self.screen, block['color'].lerp((255,255,255), 0.5), rect, width=2, border_radius=4)

        # Update and draw particles
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)
            else:
                p.draw(self.screen)
        
        # Draw push effects
        for effect in self.push_effects[:]:
            effect['life'] -= 1
            if effect['life'] <= 0:
                self.push_effects.remove(effect)
            else:
                alpha = int(100 * (effect['life'] / 5))
                color = (self.COLOR_TEXT.r, self.COLOR_TEXT.g, self.COLOR_TEXT.b, alpha)
                bar = pygame.Surface((10, self.HEIGHT) if effect['dir'] in [3,4] else (self.WIDTH, 10), pygame.SRCALPHA)
                bar.fill(color)
                if effect['dir'] == 1: self.screen.blit(bar, (0,0)) # UP
                elif effect['dir'] == 2: self.screen.blit(bar, (0, self.HEIGHT - 10)) # DOWN
                elif effect['dir'] == 3: self.screen.blit(bar, (0,0)) # LEFT
                elif effect['dir'] == 4: self.screen.blit(bar, (self.WIDTH - 10, 0)) # RIGHT

    def _render_ui(self):
        # Moves left
        moves_text = self.font_small.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (10, 10))

        # Level
        level_text = self.font_small.render(f"Level: {self.level}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.WIDTH // 2 - level_text.get_width() // 2, 10))
        
        # Score
        score_text = self.font_small.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        # Game Over / Level Complete message
        if self.game_over:
            msg = self.font_big.render("GAME OVER", True, self.BLOCK_COLORS[0])
            msg_rect = msg.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg, msg_rect)
        elif self.level_complete_msg_timer > 0:
            is_win_message = self._check_win_condition() and self.level > 1
            alpha = min(255, int(255 * (self.level_complete_msg_timer / 30.0)))
            msg_text = f"LEVEL {self.level}"
            if is_win_message:
                msg_text = f"LEVEL {self.level-1} COMPLETE!"
            
            msg_surf = self.font_big.render(msg_text, True, self.COLOR_TEXT)
            msg_surf.set_alpha(alpha)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "moves_left": self.moves_left,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
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
        
        # Test random agent survival
        self.reset()
        is_terminated = False
        for _ in range(50):
            action = self.action_space.sample()
            _, _, term, _, _ = self.step(action)
            if term:
                is_terminated = True
                break
        assert self.steps >= 50 or is_terminated, "Random agent failed to survive 50 steps"
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Create a window to display the game
    # This is not headless, so we need to unset the dummy driver
    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.display.init()
    pygame.display.set_caption("Push Block Puzzle")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    print("Game starting. Use arrow keys to push. Close window to quit.")
    print(env.user_guide)
    print(env.game_description)

    running = True
    while running:
        # Manual control
        current_action = [0, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    current_action[0] = 1
                elif event.key == pygame.K_DOWN:
                    current_action[0] = 2
                elif event.key == pygame.K_LEFT:
                    current_action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    current_action[0] = 4
                elif event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    terminated = False
                    print("--- GAME RESET ---")
        
        # Only step if an action was taken and game is not over
        if not terminated and current_action[0] != 0:
            obs, reward, term, trunc, info = env.step(current_action)
            terminated = term
            print(f"Move: {info['moves_left']}, Action: {current_action}, Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")
        else:
            # Get observation even on no-op to keep UI messages animating
            obs = env._get_observation()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    env.close()