
# Generated: 2025-08-28T02:32:23.265401
# Source Brief: brief_01735.md
# Brief Index: 1735

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    # Short, user-facing control string
    user_guide = (
        "Use arrow keys to move. Stand on a green bug and press Space to catch it. Avoid the red spiders!"
    )

    # Short, user-facing description of the game
    game_description = (
        "Hunt bugs and avoid spiders in a grid-based environment to learn about risk and reward. "
        "Collect 20 bugs to win, but 3 spider hits and you lose."
    )

    # Frames advance only when an action is received
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 16, 10
        self.CELL_W = self.WIDTH // self.GRID_COLS
        self.CELL_H = self.HEIGHT // self.GRID_ROWS
        self.MAX_STEPS = 1000
        self.WIN_BUGS = 20
        self.LOSE_HITS = 3
        self.NUM_BUGS = 5
        self.NUM_SPIDERS = 3
        self.SPIDER_START_SPEED = 0.05 # grid cells per step
        self.SPIDER_SPEED_INCREASE = 0.001
        self.DIFFICULTY_INTERVAL = 100

        # Colors
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (45, 50, 62)
        self.COLOR_PLAYER = (66, 135, 245)
        self.COLOR_PLAYER_GLOW = (120, 175, 255, 100)
        self.COLOR_BUG = (80, 220, 100)
        self.COLOR_SPIDER_BODY = (255, 50, 50)
        self.COLOR_SPIDER_LEGS = (200, 40, 40)
        self.COLOR_TEXT = (230, 230, 230)

        # EXACT spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("sans-serif", 48, bold=True)

        # Initialize state variables
        self.player_pos = None
        self.bugs = None
        self.spiders = None
        self.particles = None
        self.steps = None
        self.score = None
        self.bugs_collected = None
        self.spider_encounters = None
        self.spider_current_speed = None
        self.game_over_message = None

        self.reset()

        # Run validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.bugs_collected = 0
        self.spider_encounters = 0
        self.spider_current_speed = self.SPIDER_START_SPEED
        self.particles = []
        self.game_over_message = ""

        self._spawn_entities()
        
        return self._get_observation(), self._get_info()

    def _spawn_entities(self):
        occupied_positions = set()

        self.player_pos = pygame.Vector2(
            self.np_random.integers(0, self.GRID_COLS),
            self.np_random.integers(0, self.GRID_ROWS)
        )
        occupied_positions.add(tuple(self.player_pos))

        self.bugs = []
        for _ in range(self.NUM_BUGS):
            self._spawn_bug(occupied_positions)

        self.spiders = []
        for _ in range(self.NUM_SPIDERS):
            pos = self._get_unoccupied_pos(occupied_positions)
            occupied_positions.add(tuple(pos))
            # Each spider patrols horizontally or vertically
            patrol_dir = pygame.Vector2(1, 0) if self.np_random.random() > 0.5 else pygame.Vector2(0, 1)
            self.spiders.append({
                "pos": pygame.Vector2(pos.x, pos.y),
                "dir": patrol_dir * (1 if self.np_random.random() > 0.5 else -1),
                "patrol_axis": "x" if patrol_dir.y == 0 else "y",
            })

    def _get_unoccupied_pos(self, occupied):
        while True:
            pos = (
                self.np_random.integers(0, self.GRID_COLS),
                self.np_random.integers(0, self.GRID_ROWS)
            )
            if pos not in occupied:
                return pygame.Vector2(pos)

    def _spawn_bug(self, occupied_positions):
        bug_pos = self._get_unoccupied_pos(occupied_positions.union({tuple(b) for b in self.bugs}))
        self.bugs.append(bug_pos)

    def step(self, action):
        if self.bugs_collected >= self.WIN_BUGS or self.spider_encounters >= self.LOSE_HITS:
            # If game is over, do nothing but return current state
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0
        
        prev_player_pos = self.player_pos.copy()
        
        # 1. Update Player Position
        if movement == 1: self.player_pos.y -= 1  # Up
        elif movement == 2: self.player_pos.y += 1  # Down
        elif movement == 3: self.player_pos.x -= 1  # Left
        elif movement == 4: self.player_pos.x += 1  # Right
        
        self.player_pos.x = np.clip(self.player_pos.x, 0, self.GRID_COLS - 1)
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.GRID_ROWS - 1)

        # 2. Proximity Rewards
        if self.bugs:
            dist_before = min(b.distance_to(prev_player_pos) for b in self.bugs)
            dist_after = min(b.distance_to(self.player_pos) for b in self.bugs)
            if dist_after < dist_before:
                reward += 1  # Moved closer to a bug
        if self.spiders:
            dist_before = min(s['pos'].distance_to(prev_player_pos) for s in self.spiders)
            dist_after = min(s['pos'].distance_to(self.player_pos) for s in self.spiders)
            if dist_after < dist_before:
                reward -= 1 # Moved closer to a spider

        # 3. Handle Bug Collection
        if space_pressed:
            bug_to_remove = None
            for bug in self.bugs:
                if bug.x == self.player_pos.x and bug.y == self.player_pos.y:
                    bug_to_remove = bug
                    break
            if bug_to_remove:
                # sfx: positive chime
                self.bugs.remove(bug_to_remove)
                self.bugs_collected += 1
                self.score += 10
                reward += 10
                self._create_particles(self.player_pos, self.COLOR_BUG, 15)
                self._spawn_bug({tuple(self.player_pos)}.union({tuple(b) for b in self.bugs}))

        # 4. Update Spiders and Check Collision
        self._update_spiders()
        for spider in self.spiders:
            if int(spider['pos'].x) == self.player_pos.x and int(spider['pos'].y) == self.player_pos.y:
                # sfx: negative buzz/hit
                self.spider_encounters += 1
                self.score -= 5
                reward -= 5
                self._create_particles(self.player_pos, self.COLOR_SPIDER_BODY, 25, 2.0)
                # To prevent multiple hits in one spot, we don't move spiders after a hit this turn.
                # In a real-time game, this would be an invulnerability frame.
                break 

        # 5. Difficulty Scaling
        if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
            self.spider_current_speed += self.SPIDER_SPEED_INCREASE

        # 6. Check for Termination
        terminated = self._check_termination()
        if terminated:
            if self.bugs_collected >= self.WIN_BUGS:
                reward += 100
                self.score += 100
                self.game_over_message = "YOU WIN!"
            elif self.spider_encounters >= self.LOSE_HITS:
                reward -= 100
                self.score -= 100
                self.game_over_message = "GAME OVER"

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            terminated = True
            if not self.game_over_message:
                self.game_over_message = "TIME UP"
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_spiders(self):
        for spider in self.spiders:
            spider['pos'] += spider['dir'] * self.spider_current_speed
            # Bounce off walls
            if spider['patrol_axis'] == 'x':
                if spider['pos'].x <= 0 or spider['pos'].x >= self.GRID_COLS - 1:
                    spider['dir'].x *= -1
                    spider['pos'].x = np.clip(spider['pos'].x, 0, self.GRID_COLS - 1)
            else: # 'y'
                if spider['pos'].y <= 0 or spider['pos'].y >= self.GRID_ROWS - 1:
                    spider['dir'].y *= -1
                    spider['pos'].y = np.clip(spider['pos'].y, 0, self.GRID_ROWS - 1)

    def _check_termination(self):
        return self.bugs_collected >= self.WIN_BUGS or self.spider_encounters >= self.LOSE_HITS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid()
        self._update_and_draw_particles()
        self._draw_spiders()
        self._draw_bugs()
        self._draw_player()
        if self.game_over_message:
            self._draw_game_over_message()
            
    def _draw_grid(self):
        for x in range(0, self.WIDTH, self.CELL_W):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_H):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))

    def _draw_player(self):
        player_center_x = int((self.player_pos.x + 0.5) * self.CELL_W)
        player_center_y = int((self.player_pos.y + 0.5) * self.CELL_H)
        
        # Glow effect
        glow_radius = int(self.CELL_W * 0.6 * (1 + 0.05 * math.sin(pygame.time.get_ticks() * 0.005)))
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, glow_radius, glow_radius, glow_radius, self.COLOR_PLAYER_GLOW)
        self.screen.blit(glow_surf, (player_center_x - glow_radius, player_center_y - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Player square
        rect = pygame.Rect(self.player_pos.x * self.CELL_W, self.player_pos.y * self.CELL_H, self.CELL_W, self.CELL_H)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=4)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER_GLOW, rect, width=2, border_radius=4)

    def _draw_bugs(self):
        for bug_pos in self.bugs:
            center_x = int((bug_pos.x + 0.5) * self.CELL_W)
            center_y = int((bug_pos.y + 0.5) * self.CELL_H)
            radius = int(self.CELL_W * 0.3 * (1 + 0.1 * math.sin(pygame.time.get_ticks() * 0.008 + bug_pos.x)))
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, self.COLOR_BUG)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, self.COLOR_BUG)

    def _draw_spiders(self):
        for spider in self.spiders:
            center_x = (spider['pos'].x + 0.5) * self.CELL_W
            center_y = (spider['pos'].y + 0.5) * self.CELL_H
            body_radius = int(self.CELL_W * 0.35)

            # Color changes with proximity to player
            dist_to_player = self.player_pos.distance_to(spider['pos'])
            intensity = np.clip(1 - (dist_to_player / (self.GRID_COLS / 2)), 0.2, 1.0)
            spider_color = (
                self.COLOR_SPIDER_BODY[0],
                int(self.COLOR_SPIDER_BODY[1] * (1 - intensity)),
                int(self.COLOR_SPIDER_BODY[2] * (1 - intensity))
            )
            
            # Animated legs
            anim_time = pygame.time.get_ticks() * 0.01
            for i in range(8):
                angle = (i / 8) * 2 * math.pi + anim_time
                leg_len = body_radius * 1.5 * (1 + 0.2 * math.sin(anim_time * 2 + i))
                start_pos = (center_x, center_y)
                end_pos = (
                    center_x + math.cos(angle) * leg_len,
                    center_y + math.sin(angle) * leg_len
                )
                pygame.draw.aaline(self.screen, self.COLOR_SPIDER_LEGS, start_pos, end_pos, 2)
            
            # Body
            pygame.gfxdraw.aacircle(self.screen, int(center_x), int(center_y), body_radius, spider_color)
            pygame.gfxdraw.filled_circle(self.screen, int(center_x), int(center_y), body_radius, spider_color)

    def _create_particles(self, pos, color, count, speed_mult=1.0):
        center_x = (pos.x + 0.5) * self.CELL_W
        center_y = (pos.y + 0.5) * self.CELL_H
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2.0 * speed_mult + 1.0
            self.particles.append({
                "pos": pygame.Vector2(center_x, center_y),
                "vel": pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                "life": self.np_random.integers(20, 40),
                "color": color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # friction
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 40))
                color = (*p['color'], alpha)
                radius = int(p['life'] / 8)
                temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(radius, radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        bug_text = self.font_ui.render(f"BUGS: {self.bugs_collected}/{self.WIN_BUGS}", True, self.COLOR_TEXT)
        hits_text = self.font_ui.render(f"HITS: {self.spider_encounters}/{self.LOSE_HITS}", True, self.COLOR_TEXT)
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        
        self.screen.blit(bug_text, (10, 10))
        self.screen.blit(hits_text, (10, 35))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
    
    def _draw_game_over_message(self):
        msg_surf = self.font_msg.render(self.game_over_message, True, self.COLOR_TEXT)
        msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
        
        # Draw a semi-transparent background for the text
        bg_rect = msg_rect.inflate(40, 40)
        s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 150))
        self.screen.blit(s, bg_rect.topleft)

        self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_collected": self.bugs_collected,
            "spider_encounters": self.spider_encounters,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv()
    
    # --- Manual Play ---
    # This part is for human interaction and visualization
    
    # Re-initialize pygame for display
    pygame.display.set_caption("Bug Hunter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for manual play
    running = True
    while running:
        action = [0, 0, 0] # no-op, release space, release shift
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # Wait for a bit before the next action to make it playable
        pygame.time.wait(100) # 10 FPS for turn-based feel
        
        if terminated and keys[pygame.K_r]:
            print("Resetting game...")
            obs, info = env.reset()
            terminated = False

    env.close()