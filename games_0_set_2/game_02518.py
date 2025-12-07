
# Generated: 2025-08-27T20:36:22.289861
# Source Brief: brief_02518.md
# Brief Index: 2518

        
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
        "Controls: Use arrow keys to move the cursor. Press space to eliminate a bug."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Eliminate swarms of procedurally generated bugs in a timed grid-based strategy game."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 18)
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        
        # Game constants
        self.FPS = 30
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = 40
        self.WIN_CONDITION_BUGS = 30
        self.MAX_GAME_TIME = 60.0
        self.MAX_STEPS = 1000
        
        # Colors
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID = (30, 40, 60)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_CURSOR_OUTLINE = (0, 0, 0)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_WARN = (255, 100, 100)
        
        self.BUG_TYPES = {
            "green": {"color": (50, 220, 100), "radius": 1.5, "reward": 1, "spawn": 0},
            "blue": {"color": (80, 150, 255), "radius": 2.8, "reward": 2, "spawn": 0},
            "red": {"color": (255, 80, 80), "radius": 1.5, "reward": 3, "spawn": 2},
        }
        
        # Initialize state variables
        self.steps = None
        self.score = None
        self.game_over = None
        self.bugs_eliminated = None
        self.game_timer = None
        self.bugs = None
        self.particles = None
        self.cursor_pos = None
        self.space_was_held = None
        self.spawn_timer = None
        self.spawn_rate = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.bugs_eliminated = 0
        self.game_timer = self.MAX_GAME_TIME
        
        self.bugs = []
        self.particles = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.space_was_held = False
        
        self.spawn_timer = 0.0
        self.spawn_rate = 0.5  # bugs per second
        
        for _ in range(5):
            self._spawn_bug()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        delta_time = 1.0 / self.FPS
        self.steps += 1
        self.game_timer -= delta_time
        
        reward = 0.0
        
        # 1. Handle player actions
        self._handle_input(action)
        
        # 2. Process click event
        click_reward = self._process_click(action)
        reward += click_reward

        # 3. Update game state
        self._update_spawner(delta_time)
        self._update_bugs(delta_time)
        self._update_particles(delta_time)
        
        # 4. Check for termination conditions
        terminated = False
        if self.bugs_eliminated >= self.WIN_CONDITION_BUGS:
            reward += 50
            terminated = True
            self.game_over = True
        elif self.game_timer <= 0:
            reward = -100
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            reward = -10
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

    def _handle_input(self, action):
        movement = action[0]
        
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

    def _process_click(self, action):
        space_pressed = action[1] == 1
        click_reward = 0
        
        if space_pressed and not self.space_was_held:
            # Sound placeholder: # sfx_click.play()
            clicked_bug_index = -1
            for i, bug in enumerate(self.bugs):
                if bug['pos'][0] == self.cursor_pos[0] and bug['pos'][1] == self.cursor_pos[1]:
                    clicked_bug_index = i
                    break
            
            if clicked_bug_index != -1:
                click_reward = self._trigger_explosion(clicked_bug_index)
        
        self.space_was_held = space_pressed
        return click_reward

    def _trigger_explosion(self, start_bug_index):
        # Sound placeholder: # sfx_explosion_start.play()
        
        q = [start_bug_index]
        processed_indices = {start_bug_index}
        bugs_to_remove = set()
        total_reward = 0

        while q:
            current_index = q.pop(0)
            
            if current_index >= len(self.bugs): continue
            current_bug = self.bugs[current_index]
            
            bugs_to_remove.add(current_index)
            total_reward += self.BUG_TYPES[current_bug['type']]['reward']
            
            # Create particles for the explosion
            self._create_particles(current_bug)
            
            # Chain reaction
            explosion_radius = self.BUG_TYPES[current_bug['type']]['radius']
            for i, other_bug in enumerate(self.bugs):
                if i not in processed_indices:
                    dist = math.hypot(current_bug['pos'][0] - other_bug['pos'][0], 
                                     current_bug['pos'][1] - other_bug['pos'][1])
                    if dist < explosion_radius:
                        q.append(i)
                        processed_indices.add(i)

        # Handle spawning from Red bugs after calculating all chain reactions
        bugs_to_spawn = 0
        for index in bugs_to_remove:
            if self.bugs[index]['type'] == 'red':
                bugs_to_spawn += self.BUG_TYPES['red']['spawn']

        # Remove exploded bugs
        self.bugs_eliminated += len(bugs_to_remove)
        self.bugs = [bug for i, bug in enumerate(self.bugs) if i not in bugs_to_remove]
        
        # Spawn new bugs from red explosions
        for _ in range(bugs_to_spawn):
            self._spawn_bug(bug_type='green')
            
        return total_reward

    def _create_particles(self, bug):
        bug_info = self.BUG_TYPES[bug['type']]
        center_x = (bug['pos'][0] + 0.5) * self.CELL_SIZE
        center_y = (bug['pos'][1] + 0.5) * self.CELL_SIZE
        
        num_particles = int(bug_info['radius'] * 20)
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4) * bug_info['radius']
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.uniform(0.5, 1.2)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'life': life,
                'max_life': life,
                'color': bug_info['color'],
                'size': self.np_random.uniform(2, 5)
            })

    def _spawn_bug(self, bug_type=None):
        if len(self.bugs) >= self.GRID_WIDTH * self.GRID_HEIGHT:
            return

        if bug_type is None:
            # Weighted random choice for bug type
            rand_val = self.np_random.random()
            if rand_val < 0.6: bug_type = "green"
            elif rand_val < 0.9: bug_type = "blue"
            else: bug_type = "red"
        
        occupied_cells = {tuple(bug['pos']) for bug in self.bugs}
        
        for _ in range(100): # Try 100 times to find an empty spot
            pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT)]
            if tuple(pos) not in occupied_cells:
                self.bugs.append({
                    'pos': pos,
                    'type': bug_type,
                    'anim_offset': self.np_random.uniform(0, 2 * math.pi)
                })
                return

    def _update_spawner(self, delta_time):
        self.spawn_rate += 0.02 * delta_time
        self.spawn_timer += delta_time
        if self.spawn_timer > 1.0 / self.spawn_rate:
            self._spawn_bug()
            self.spawn_timer = 0

    def _update_bugs(self, delta_time):
        # This is where bug-specific animations would go.
        # Currently, the pulsing is handled in the render function based on game time.
        pass

    def _update_particles(self, delta_time):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= delta_time
        self.particles = [p for p in self.particles if p['life'] > 0]
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
            
        # Draw particles
        for p in self.particles:
            life_ratio = max(0, p['life'] / p['max_life'])
            alpha = int(255 * life_ratio)
            color = p['color']
            
            # Using a temporary surface for alpha blending
            s = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (color[0], color[1], color[2], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(s, (int(p['pos'][0] - p['size']), int(p['pos'][1] - p['size'])))

        # Draw bugs
        anim_time = self.steps * 0.1
        for bug in self.bugs:
            center_x = int((bug['pos'][0] + 0.5) * self.CELL_SIZE)
            center_y = int((bug['pos'][1] + 0.5) * self.CELL_SIZE)
            
            pulse = (math.sin(anim_time + bug['anim_offset']) + 1) / 2
            radius = int(self.CELL_SIZE * 0.3 + pulse * 3)
            
            color = self.BUG_TYPES[bug['type']]['color']
            outline_color = tuple(min(255, c + 50) for c in color)
            
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius + 1, outline_color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)

        # Draw cursor
        cursor_px = int((self.cursor_pos[0] + 0.5) * self.CELL_SIZE)
        cursor_py = int((self.cursor_pos[1] + 0.5) * self.CELL_SIZE)
        size = self.CELL_SIZE // 4
        
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (cursor_px - size - 1, cursor_py), (cursor_px + size + 1, cursor_py), 5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR_OUTLINE, (cursor_px, cursor_py - size - 1), (cursor_px, cursor_py + size + 1), 5)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_px - size, cursor_py), (cursor_px + size, cursor_py), 3)
        pygame.draw.line(self.screen, self.COLOR_CURSOR, (cursor_px, cursor_py - size), (cursor_px, cursor_py + size), 3)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))

        # Timer
        timer_color = self.COLOR_TEXT if self.game_timer > 10 else self.COLOR_TEXT_WARN
        timer_text = self.font_small.render(f"TIME: {max(0, int(self.game_timer)) + 1:02d}", True, timer_color)
        self.screen.blit(timer_text, (10, 10))

        # Bug Count
        bug_count_text = self.font_small.render(f"BUGS: {self.bugs_eliminated}/{self.WIN_CONDITION_BUGS}", True, self.COLOR_TEXT)
        self.screen.blit(bug_count_text, (10, 30))
        
        # Game Over / Win Text
        if self.game_over:
            if self.bugs_eliminated >= self.WIN_CONDITION_BUGS:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            
            # Draw a semi-transparent background for the text
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_eliminated": self.bugs_eliminated,
            "timer": self.game_timer,
        }

    def close(self):
        pygame.font.quit()
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Override auto_advance for human play
    env.auto_advance = False 
    
    running = True
    terminated = False
    
    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Bug Swarm")
    game_clock = pygame.time.Clock()

    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward}, Score: {info['score']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            # Wait a bit on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        game_clock.tick(env.FPS)

    env.close()