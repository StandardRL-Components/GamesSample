
# Generated: 2025-08-28T04:59:04.570068
# Source Brief: brief_05428.md
# Brief Index: 5428

        
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

    # Short, user-facing control string
    user_guide = (
        "Controls: Arrow keys to move your character. Push crates onto the green targets."
    )

    # Short, user-facing description of the game
    game_description = (
        "Race against time to solve a Sokoban puzzle by pushing crates onto target locations within 30 seconds."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    GRID_SIZE = 16, 10
    CELL_SIZE = 40

    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (50, 50, 60)
    COLOR_WALL = (10, 10, 15)
    COLOR_PLAYER = (255, 80, 80)
    COLOR_PLAYER_GLOW = (255, 120, 120)
    COLOR_CRATE = (160, 110, 70)
    COLOR_CRATE_GLOW = (190, 140, 100)
    COLOR_TARGET = (60, 150, 60)
    COLOR_TARGET_ACTIVE = (80, 220, 80)
    COLOR_TEXT = (220, 220, 230)
    COLOR_FLASH = (255, 255, 255)
    
    # Game mechanics
    TIME_LIMIT_SECONDS = 30
    MOVE_COOLDOWN_FRAMES = 5  # Cooldown to make movement feel deliberate

    class Particle:
        """A simple particle for visual effects."""
        def __init__(self, pos, vel, radius, color, lifespan):
            self.pos = list(pos)
            self.vel = list(vel)
            self.radius = radius
            self.color = color
            self.lifespan = lifespan
            self.max_lifespan = lifespan

        def update(self):
            self.pos[0] += self.vel[0]
            self.pos[1] += self.vel[1]
            self.lifespan -= 1
            self.radius -= 0.1

        def draw(self, surface):
            if self.lifespan > 0 and self.radius > 0:
                alpha = int(255 * (self.lifespan / self.max_lifespan))
                r, g, b = self.color
                pygame.gfxdraw.filled_circle(
                    surface, int(self.pos[0]), int(self.pos[1]), int(self.radius), (r, g, b, alpha)
                )

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup for headless rendering
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Level layout
        self.level_layout = [
            "WWWWWWWWWWWWWWWW",
            "W      WWWW    W",
            "W      W  T    W",
            "W      W C WWWWW",
            "W  WWWWW C T   W",
            "W  W   W C W   W",
            "W  T P   W W   W",
            "W  WWWWWWWWWW  W",
            "W              W",
            "WWWWWWWWWWWWWWWW",
        ]
        
        # Initialize state variables
        self.player_pos = (0, 0)
        self.crates_pos = []
        self.targets_pos = []
        self.walls = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = self.TIME_LIMIT_SECONDS * self.FPS
        self.move_cooldown = 0
        self.particles = []
        self.flash_timers = {} # {(x, y): timer}
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.move_cooldown = 0
        self.particles = []
        self.flash_timers = {}

        self.crates_pos = []
        self.targets_pos = []
        self.walls = []

        for y, row in enumerate(self.level_layout):
            for x, char in enumerate(row):
                pos = (x, y)
                if char == 'W':
                    self.walls.append(pos)
                elif char == 'P':
                    self.player_pos = pos
                elif char == 'C':
                    self.crates_pos.append(pos)
                elif char == 'T':
                    self.targets_pos.append(pos)
        
        self._initial_crates_pos = list(self.crates_pos)
        self._initial_player_pos = self.player_pos
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Time penalty
        
        # --- Update Cooldowns and Effects ---
        self.move_cooldown = max(0, self.move_cooldown - 1)
        self._update_particles()
        self._update_flash_timers()

        # --- Process Action ---
        movement = action[0]
        
        if movement != 0 and self.move_cooldown == 0:
            self.move_cooldown = self.MOVE_COOLDOWN_FRAMES
            
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(movement, (0, 0))
            
            next_player_pos = (self.player_pos[0] + dx, self.player_pos[1] + dy)
            
            # Store pre-move state for reward calculation
            crates_before_move = list(self.crates_pos)
            crates_on_target_before = {c for c in crates_before_move if c in self.targets_pos}

            # --- Collision and Movement Logic ---
            if next_player_pos in self.walls:
                pass # Player hits a wall
            elif next_player_pos in self.crates_pos:
                # Player attempts to push a crate
                crate_idx = self.crates_pos.index(next_player_pos)
                pushed_crate_pos = self.crates_pos[crate_idx]
                next_crate_pos = (pushed_crate_pos[0] + dx, pushed_crate_pos[1] + dy)

                if next_crate_pos not in self.walls and next_crate_pos not in self.crates_pos:
                    # Valid push
                    # SFX: Crate slide
                    self._create_particles(self.player_pos, self.COLOR_PLAYER, 5)
                    self.player_pos = next_player_pos
                    
                    # Calculate distance-based reward for the pushed crate
                    reward += self._calculate_crate_move_reward(pushed_crate_pos, next_crate_pos)
                    
                    self.crates_pos[crate_idx] = next_crate_pos
                    self.flash_timers[next_crate_pos] = 5 # Trigger flash effect
            else:
                # Player moves into an empty space
                # SFX: Player step
                self._create_particles(self.player_pos, self.COLOR_PLAYER, 3)
                self.player_pos = next_player_pos

            # Calculate event-based reward for crates on targets
            crates_on_target_after = {c for c in self.crates_pos if c in self.targets_pos}
            newly_on_target = crates_on_target_after - crates_on_target_before
            if newly_on_target:
                # SFX: Target activated
                reward += len(newly_on_target) * 10.0

        # --- Update Game State ---
        self.steps += 1
        self.score += reward
        terminated = self._check_termination()
        
        # Apply terminal rewards
        if terminated:
            if all(c in self.targets_pos for c in self.crates_pos): # Win condition
                # SFX: Level complete
                reward += 50.0
            else: # Timeout
                # SFX: Game over
                reward -= 100.0
            self.score += reward # Add terminal reward to final score
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _calculate_crate_move_reward(self, old_pos, new_pos):
        unoccupied_targets = [t for t in self.targets_pos if t not in self.crates_pos or t == old_pos]
        if not unoccupied_targets:
            return 0

        def manhattan(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        dist_before = min(manhattan(old_pos, t) for t in unoccupied_targets)
        dist_after = min(manhattan(new_pos, t) for t in unoccupied_targets)

        if dist_after < dist_before:
            return 1.0
        elif dist_after > dist_before:
            return -2.0
        return 0

    def _check_termination(self):
        if self.game_over:
            return True
        
        win = all(c in self.targets_pos for c in self.crates_pos)
        timeout = self.steps >= self.max_steps
        
        if win or timeout:
            self.game_over = True
            return True
        return False
        
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_grid_and_walls()
        self._render_targets()
        self._render_entities()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _to_pixels(self, grid_pos):
        gx, gy = grid_pos
        offset_x = (self.WIDTH - self.GRID_SIZE[0] * self.CELL_SIZE) // 2
        offset_y = (self.HEIGHT - self.GRID_SIZE[1] * self.CELL_SIZE) // 2
        return (
            offset_x + gx * self.CELL_SIZE,
            offset_y + gy * self.CELL_SIZE
        )

    def _render_grid_and_walls(self):
        offset_x = (self.WIDTH - self.GRID_SIZE[0] * self.CELL_SIZE) // 2
        offset_y = (self.HEIGHT - self.GRID_SIZE[1] * self.CELL_SIZE) // 2
        
        for y in range(self.GRID_SIZE[1] + 1):
            start = (offset_x, offset_y + y * self.CELL_SIZE)
            end = (offset_x + self.GRID_SIZE[0] * self.CELL_SIZE, offset_y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_SIZE[0] + 1):
            start = (offset_x + x * self.CELL_SIZE, offset_y)
            end = (offset_x + x * self.CELL_SIZE, offset_y + self.GRID_SIZE[1] * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
            
        for wall_pos in self.walls:
            px, py = self._to_pixels(wall_pos)
            pygame.draw.rect(self.screen, self.COLOR_WALL, (px, py, self.CELL_SIZE, self.CELL_SIZE))

    def _render_targets(self):
        for target_pos in self.targets_pos:
            px, py = self._to_pixels(target_pos)
            is_active = target_pos in self.crates_pos
            color = self.COLOR_TARGET_ACTIVE if is_active else self.COLOR_TARGET
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE))

    def _render_entities(self):
        # Render crates
        for crate_pos in self.crates_pos:
            px, py = self._to_pixels(crate_pos)
            rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            # Glow for interaction readiness
            glow_rect = rect.inflate(6, 6)
            pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_CRATE_GLOW, 50))

            # Main crate body
            pygame.draw.rect(self.screen, self.COLOR_CRATE, rect, border_radius=4)
            
            # Flash effect on push
            if crate_pos in self.flash_timers:
                alpha = int(255 * (self.flash_timers[crate_pos] / 5.0))
                flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(flash_surface, (*self.COLOR_FLASH, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=4)
                self.screen.blit(flash_surface, (px, py))

        # Render player
        px, py = self._to_pixels(self.player_pos)
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        # Glow effect
        glow_rect = rect.inflate(8, 8)
        pygame.gfxdraw.box(self.screen, glow_rect, (*self.COLOR_PLAYER_GLOW, 80))
        
        # Main player body
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, rect, border_radius=6)
        
    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)
            
    def _render_ui(self):
        # Crates on target count
        num_on_target = sum(1 for c in self.crates_pos if c in self.targets_pos)
        crates_text = f"CRATES: {num_on_target} / {len(self.targets_pos)}"
        text_surf = self.font_small.render(crates_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Timer
        time_left = max(0, (self.max_steps - self.steps) / self.FPS)
        timer_text = f"TIME: {time_left:.1f}s"
        text_surf = self.font_small.render(timer_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(topright=(self.WIDTH - 10, 10))
        self.screen.blit(text_surf, text_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = all(c in self.targets_pos for c in self.crates_pos)
            msg = "YOU WIN!" if win else "TIME'S UP!"
            color = self.COLOR_TARGET_ACTIVE if win else self.COLOR_PLAYER
            
            text_surf = self.font_large.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "crates_on_target": sum(1 for c in self.crates_pos if c in self.targets_pos),
            "time_remaining_steps": self.max_steps - self.steps,
        }

    def _create_particles(self, grid_pos, color, count):
        px, py = self._to_pixels(grid_pos)
        center_x = px + self.CELL_SIZE // 2
        center_y = py + self.CELL_SIZE // 2
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 1.5)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            radius = random.uniform(2, 5)
            lifespan = random.randint(10, 20)
            self.particles.append(self.Particle((center_x, center_y), vel, radius, color, lifespan))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            
    def _update_flash_timers(self):
        self.flash_timers = {pos: timer - 1 for pos, timer in self.flash_timers.items() if timer > 1}

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
    # This block allows you to play the game manually
    # It will not be executed when the environment is used by an RL agent
    
    # Set up Pygame display
    pygame.display.set_caption("Sokoban Arcade")
    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    while running:
        movement_action = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2
        elif keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
            
        space_action = 1 if keys[pygame.K_SPACE] else 0
        shift_action = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement_action, space_action, shift_action]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(env.FPS)
        
    pygame.quit()