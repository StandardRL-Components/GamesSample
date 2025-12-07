
# Generated: 2025-08-28T04:37:20.443561
# Source Brief: brief_02367.md
# Brief Index: 2367

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move. Press space to stomp and squash nearby bugs."
    )

    game_description = (
        "An isometric arcade game where you must squash all the bugs before time runs out. "
        "Different colored bugs are worth different points. Stomp strategically to clear them all!"
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_OBSTACLE = (80, 90, 100)
    COLOR_PLAYER = (50, 255, 150)
    COLOR_PLAYER_GLOW = (150, 255, 200)
    BUG_COLORS = {
        1: ((0, 200, 0), (100, 255, 100)),  # Green (body, glow)
        2: ((0, 150, 255), (100, 200, 255)), # Blue
        3: ((255, 50, 50), (255, 150, 150)),   # Red
    }
    COLOR_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 100)
    
    # Screen and Grid
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 20
    GRID_HEIGHT = 15
    TILE_WIDTH_HALF = 18
    TILE_HEIGHT_HALF = 9
    
    # Game Parameters
    FPS = 30
    TIME_LIMIT_SECONDS = 60
    TOTAL_BUGS = 30
    OBSTACLE_COUNT = 25
    PLAYER_SPEED = 3.0
    STOMP_COOLDOWN_FRAMES = 15 # 0.5 seconds
    STOMP_RADIUS = 1.5 # Grid units

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        self.grid_origin_x = self.SCREEN_WIDTH // 2
        self.grid_origin_y = 60

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_grid_pos = None
        self.grid = None
        self.bugs = None
        self.particles = None
        self.stomp_cooldown = None
        self.stomp_active_frame = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.timer_frames = None
        self.np_random = None

        self.reset()
        # self.validate_implementation() # Commented out for final submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer_frames = self.TIME_LIMIT_SECONDS * self.FPS
        self.stomp_cooldown = 0
        self.stomp_active_frame = -1

        self.particles = []
        self._place_entities()
        
        return self._get_observation(), self._get_info()
    
    def _place_entities(self):
        # 1. Create grid and place obstacles
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Place player
        self.player_grid_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.player_pos = np.array(self._iso_to_cart(self.player_grid_pos[0], self.player_grid_pos[1]), dtype=float)

        # Place obstacles
        obstacle_candidates = list(np.ndindex(self.grid.shape))
        obstacle_candidates.remove(self.player_grid_pos)
        self.np_random.shuffle(obstacle_candidates)
        for i in range(self.OBSTACLE_COUNT):
            if i < len(obstacle_candidates):
                ox, oy = obstacle_candidates[i]
                self.grid[ox, oy] = 1 # 1 represents an obstacle

        # 2. Determine reachable tiles using BFS
        q = deque([self.player_grid_pos])
        reachable = {self.player_grid_pos}
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and \
                   self.grid[nx, ny] == 0 and (nx, ny) not in reachable:
                    reachable.add((nx, ny))
                    q.append((nx, ny))
        
        # 3. Place bugs on reachable, non-player tiles
        self.bugs = []
        bug_spawn_points = list(reachable - {self.player_grid_pos})
        self.np_random.shuffle(bug_spawn_points)
        
        for i in range(self.TOTAL_BUGS):
            if i < len(bug_spawn_points):
                bx, by = bug_spawn_points[i]
                bug_type = self.np_random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                self.bugs.append({
                    "grid_pos": (bx, by),
                    "type": bug_type,
                    "pulse": self.np_random.random() * 2 * math.pi
                })

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = -0.01  # Time penalty

        # --- Update Game Logic ---
        self._update_player(movement)
        
        stomp_reward = self._update_stomp(space_pressed)
        reward += stomp_reward

        self._update_particles()
        self._update_bugs()
        
        self.timer_frames -= 1
        self.steps += 1
        
        # --- Check Termination ---
        terminated = False
        if len(self.bugs) == 0:
            reward += 50
            terminated = True
            self.game_over = True
        elif self.timer_frames <= 0:
            reward += -50
            terminated = True
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_player(self, movement):
        move_vec = np.array([0.0, 0.0])
        if movement == 1: # Up (Iso Up-Left)
            move_vec = np.array([-1.0, -0.5])
        elif movement == 2: # Down (Iso Down-Right)
            move_vec = np.array([1.0, 0.5])
        elif movement == 3: # Left (Iso Down-Left)
            move_vec = np.array([-1.0, 0.5])
        elif movement == 4: # Right (Iso Up-Right)
            move_vec = np.array([1.0, -0.5])

        if np.linalg.norm(move_vec) > 0:
            move_vec = move_vec / np.linalg.norm(move_vec)
        
        new_pos = self.player_pos + move_vec * self.PLAYER_SPEED
        
        # Collision detection
        new_grid_pos = self._cart_to_iso(new_pos[0], new_pos[1])
        if 0 <= new_grid_pos[0] < self.GRID_WIDTH and 0 <= new_grid_pos[1] < self.GRID_HEIGHT:
            if self.grid[new_grid_pos[0], new_grid_pos[1]] == 0:
                self.player_pos = new_pos
            else: # Slide along wall
                # Try moving only on X axis
                pos_x = self.player_pos + np.array([move_vec[0] * self.PLAYER_SPEED, 0])
                grid_pos_x = self._cart_to_iso(pos_x[0], pos_x[1])
                if 0 <= grid_pos_x[0] < self.GRID_WIDTH and 0 <= grid_pos_x[1] < self.GRID_HEIGHT and self.grid[grid_pos_x[0], grid_pos_x[1]] == 0:
                    self.player_pos = pos_x
                else:
                    # Try moving only on Y axis
                    pos_y = self.player_pos + np.array([0, move_vec[1] * self.PLAYER_SPEED])
                    grid_pos_y = self._cart_to_iso(pos_y[0], pos_y[1])
                    if 0 <= grid_pos_y[0] < self.GRID_WIDTH and 0 <= grid_pos_y[1] < self.GRID_HEIGHT and self.grid[grid_pos_y[0], grid_pos_y[1]] == 0:
                        self.player_pos = pos_y

        self.player_grid_pos = self._cart_to_iso(self.player_pos[0], self.player_pos[1])

    def _update_stomp(self, space_pressed):
        reward = 0
        if self.stomp_cooldown > 0:
            self.stomp_cooldown -= 1
        
        if space_pressed and self.stomp_cooldown == 0:
            # Sfx: Stomp sound
            self.stomp_cooldown = self.STOMP_COOLDOWN_FRAMES
            self.stomp_active_frame = self.steps
            
            bugs_to_remove = []
            player_gx, player_gy = self.player_grid_pos
            
            for bug in self.bugs:
                bug_gx, bug_gy = bug["grid_pos"]
                dist = math.sqrt((player_gx - bug_gx)**2 + (player_gy - bug_gy)**2)
                if dist <= self.STOMP_RADIUS:
                    bugs_to_remove.append(bug)
            
            for bug in bugs_to_remove:
                # Sfx: Bug squash sound
                self.bugs.remove(bug)
                reward += bug["type"]
                self.score += bug["type"]
                self._add_particles(bug["grid_pos"], bug["type"])
        
        return reward

    def _update_bugs(self):
        for bug in self.bugs:
            bug["pulse"] += 0.2

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            if p["lifespan"] <= 0:
                self.particles.remove(p)

    def _add_particles(self, grid_pos, bug_type):
        cart_pos = self._iso_to_cart(grid_pos[0] + 0.5, grid_pos[1] + 0.5)
        color = self.BUG_COLORS[bug_type][0]
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 2 + 1
            self.particles.append({
                "pos": list(cart_pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(10, 20),
                "color": color,
                "radius": self.np_random.random() * 2 + 1
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render grid
        for y in range(self.GRID_HEIGHT + 1):
            p1 = self._iso_to_cart(0, y)
            p2 = self._iso_to_cart(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
        for x in range(self.GRID_WIDTH + 1):
            p1 = self._iso_to_cart(x, 0)
            p2 = self._iso_to_cart(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)

        # Render obstacles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] == 1:
                    p1 = self._iso_to_cart(x, y)
                    p2 = self._iso_to_cart(x + 1, y)
                    p3 = self._iso_to_cart(x + 1, y + 1)
                    p4 = self._iso_to_cart(x, y + 1)
                    pygame.gfxdraw.filled_polygon(self.screen, [p1,p2,p3,p4], self.COLOR_OBSTACLE)

        # Combine bugs and player for correct Z-ordering
        render_list = []
        for bug in self.bugs:
            render_list.append(('bug', bug, bug['grid_pos'][0] + bug['grid_pos'][1]))
        
        player_z = self.player_grid_pos[0] + self.player_grid_pos[1] + 0.5
        render_list.append(('player', None, player_z))

        render_list.sort(key=lambda item: item[2])

        for item_type, item_data, _ in render_list:
            if item_type == 'bug':
                cart_pos = self._iso_to_cart(item_data["grid_pos"][0] + 0.5, item_data["grid_pos"][1] + 0.5)
                body_color, glow_color = self.BUG_COLORS[item_data["type"]]
                radius = 4 + math.sin(item_data["pulse"])
                glow_radius = radius * 1.8
                
                # Using a separate surface for the glow for better alpha blending
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*glow_color, 80), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (cart_pos[0] - glow_radius, cart_pos[1] - glow_radius))

                pygame.gfxdraw.filled_circle(self.screen, int(cart_pos[0]), int(cart_pos[1]), int(radius), body_color)
                pygame.gfxdraw.aacircle(self.screen, int(cart_pos[0]), int(cart_pos[1]), int(radius), body_color)

            elif item_type == 'player':
                player_radius = 6
                glow_radius = player_radius + 4
                
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*self.COLOR_PLAYER_GLOW, 100), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (self.player_pos[0] - glow_radius, self.player_pos[1] - glow_radius))

                pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), player_radius, self.COLOR_PLAYER)
                pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), player_radius, self.COLOR_PLAYER)

        # Render stomp effect on top
        if self.stomp_active_frame != -1 and self.steps - self.stomp_active_frame < 10:
            progress = (self.steps - self.stomp_active_frame) / 10.0
            radius = progress * self.STOMP_RADIUS * self.TILE_WIDTH_HALF * 1.5
            alpha = int(200 * (1 - progress))
            if alpha > 0:
                s = pygame.Surface((int(radius*2), int(radius*2)), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 255, alpha), (radius, radius), radius, width=max(1, int(5 * (1-progress))))
                self.screen.blit(s, (self.player_pos[0] - radius, self.player_pos[1] - radius))
        
        # Render particles on top
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 20.0))
            color = (*p["color"], alpha)
            pygame.draw.circle(self.screen, color, p["pos"], p["radius"])

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Bugs remaining
        bugs_text = self.font_small.render(f"BUGS: {len(self.bugs)}/{self.TOTAL_BUGS}", True, self.COLOR_TEXT)
        self.screen.blit(bugs_text, (10, 40))

        # Timer
        time_left_sec = max(0, self.timer_frames / self.FPS)
        timer_color = self.COLOR_TEXT if time_left_sec > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_large.render(f"{time_left_sec:.1f}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if len(self.bugs) == 0 else "TIME'S UP!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 - 20))
            self.screen.blit(end_text, end_rect)

            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "bugs_remaining": len(self.bugs),
            "time_left": max(0, self.timer_frames / self.FPS),
        }

    # --- Helper Functions ---
    def _iso_to_cart(self, iso_x, iso_y):
        cart_x = (iso_x - iso_y) * self.TILE_WIDTH_HALF + self.grid_origin_x
        cart_y = (iso_x + iso_y) * self.TILE_HEIGHT_HALF + self.grid_origin_y
        return cart_x, cart_y

    def _cart_to_iso(self, cart_x, cart_y):
        cart_x_shifted = cart_x - self.grid_origin_x
        cart_y_shifted = cart_y - self.grid_origin_y
        iso_x = (cart_x_shifted / self.TILE_WIDTH_HALF + cart_y_shifted / self.TILE_HEIGHT_HALF) / 2
        iso_y = (cart_y_shifted / self.TILE_HEIGHT_HALF - cart_x_shifted / self.TILE_WIDTH_HALF) / 2
        return int(round(iso_x)), int(round(iso_y))

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    import os
    # Set the video driver to a dummy one if you are running this in a headless environment
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        pass
    else:
        # If not headless, we can show the screen
        GameEnv.metadata["render_modes"].append("human")
        
    env = GameEnv()
    obs, info = env.reset()
    
    # --- Human player setup ---
    pygame.display.set_caption("Bug Squasher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    terminated = False
    while True:
        # --- Get human input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_pressed = 1 if keys[pygame.K_SPACE] else 0
        shift_pressed = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_pressed, shift_pressed]

        # --- Check for quit ---
        should_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                should_quit = True
        if should_quit:
            break

        # --- Step environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            pygame.time.wait(2000) # Wait 2 seconds before resetting
            obs, info = env.reset()
            terminated = False

    env.close()