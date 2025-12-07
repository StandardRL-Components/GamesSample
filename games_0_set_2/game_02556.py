import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# Set headless mode for Pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A Gymnasium environment for a haunted house escape game.

    The player must navigate a procedurally generated house, collect artifacts,
    and reach the exit while avoiding patrolling ghosts.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing control string
    user_guide = (
        "Controls: Arrow keys to move. Hold Shift to sprint. Press Space to collect artifacts."
    )

    # User-facing description of the game
    game_description = (
        "Escape a procedurally generated haunted house. Collect yellow artifacts for points, "
        "reach the green exit to win, and avoid the red ghosts!"
    )

    # The state is static until a user submits an action.
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 1000
        self.N_GHOSTS = 5
        self.N_ITEMS = 5
        self.GHOST_CATCH_LIMIT = 5
        self.PLAYER_SPEED = 3
        self.SPRINT_MULTIPLIER = 1.5
        self.GHOST_SPEED = 1

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # Colors
        self.COLOR_BG = (10, 5, 15)
        self.COLOR_WALL = (40, 30, 50)
        self.COLOR_FLOOR = (60, 50, 70)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_PLAYER_GLOW = (0, 128, 128)
        self.COLOR_GHOST = (255, 0, 0)
        self.COLOR_GHOST_GLOW = (128, 0, 0)
        self.COLOR_ITEM = (255, 255, 0)
        self.COLOR_EXIT = (0, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_VIGNETTE = (0, 0, 0)

        # State variables are initialized in reset()
        self.player_pos = None
        self.ghosts = []
        self.items = []
        self.exit_rect = None
        self.walls = []
        self.floors = []
        self.steps = 0
        self.score = 0
        self.ghost_catches = 0
        self.game_over = False
        self.game_won = False
        self.np_random = None
        self.particles = []

        # This will be created once and reused
        self._vignette_surface = self._create_vignette()

    def _create_vignette(self):
        """Creates a vignette surface to darken the screen edges."""
        vignette = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        radius = int(self.HEIGHT * 0.8)
        for i in range(radius, 0, -1):
            alpha = int(255 * (1 - (i / radius)**2))
            alpha = min(255, max(0, alpha))
            pygame.gfxdraw.filled_circle(
                vignette, self.WIDTH // 2, self.HEIGHT // 2, i, (*self.COLOR_VIGNETTE, alpha)
            )
        # Invert alpha
        vignette_inv = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        vignette_inv.fill((0,0,0,200))
        vignette_inv.blit(vignette, (0,0), special_flags=pygame.BLEND_RGBA_SUB)
        return vignette_inv

    def _generate_level(self):
        """Procedurally generates the house layout, items, and entity positions."""
        self.walls, self.floors = [], []
        grid_w, grid_h = self.WIDTH // 10, self.HEIGHT // 10
        grid = np.ones((grid_h, grid_w))  # 1 is wall, 0 is floor

        rooms = []
        for _ in range(10): # Attempt to place 10 rooms
            w = self.np_random.integers(5, 10)
            h = self.np_random.integers(4, 8)
            x = self.np_random.integers(1, grid_w - w - 1)
            y = self.np_random.integers(1, grid_h - h - 1)
            new_room = pygame.Rect(x, y, w, h)
            
            # Check for overlap
            if not any(new_room.colliderect(r) for r in rooms):
                rooms.append(new_room)
                grid[y:y+h, x:x+w] = 0

        # Connect rooms
        if len(rooms) > 1:
            for i in range(len(rooms) - 1):
                x1, y1 = rooms[i].centerx, rooms[i].centery
                x2, y2 = rooms[i+1].centerx, rooms[i+1].centery
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    if 0 <= y1 < grid_h and 0 <= x < grid_w:
                        grid[y1, x] = 0
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= y < grid_h and 0 <= x2 < grid_w:
                        grid[y, x2] = 0

        # Create rects from grid
        for r in range(grid_h):
            for c in range(grid_w):
                rect = pygame.Rect(c * 10, r * 10, 10, 10)
                if grid[r, c] == 1:
                    self.walls.append(rect)
                else:
                    self.floors.append(rect)
        
        # Ensure level is playable
        if not self.floors: # Failsafe for empty level
            self.floors.append(pygame.Rect(0,0,self.WIDTH, self.HEIGHT))

        walkable_tiles = self.floors

        # Place player
        start_pos_rect = walkable_tiles[self.np_random.integers(len(walkable_tiles))]
        self.player_pos = pygame.Vector2(start_pos_rect.center)

        # Place exit far from player
        max_dist = 0
        exit_tile = walkable_tiles[0]
        for tile in walkable_tiles:
            dist = self.player_pos.distance_to(tile.center)
            if dist > max_dist:
                max_dist = dist
                exit_tile = tile
        self.exit_rect = pygame.Rect(exit_tile.topleft, (20, 20))
        self.exit_rect.center = exit_tile.center

        # Place items
        self.items = []
        for _ in range(self.N_ITEMS):
            item_tile = walkable_tiles[self.np_random.integers(len(walkable_tiles))]
            item_rect = pygame.Rect(0, 0, 10, 10)
            item_rect.center = item_tile.center
            if not item_rect.colliderect(self.exit_rect):
                self.items.append(item_rect)

        # Place ghosts
        self.ghosts = []
        for _ in range(self.N_GHOSTS):
            start_tile = walkable_tiles[self.np_random.integers(len(walkable_tiles))]
            ghost = {
                "pos": pygame.Vector2(start_tile.center),
                "waypoints": [pygame.Vector2(walkable_tiles[self.np_random.integers(len(walkable_tiles))].center) for _ in range(4)],
                "waypoint_idx": 0,
                "bob_offset": self.np_random.uniform(0, math.pi * 2)
            }
            self.ghosts.append(ghost)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.ghost_catches = 0
        self.game_over = False
        self.game_won = False
        self.particles = []

        self._generate_level()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Store state for reward calculation
        prev_player_pos = self.player_pos.copy()
        
        # Update game logic
        self._handle_player_movement(movement, shift_held)
        self._update_ghosts()
        
        # Handle interactions and collisions
        reward_from_events = self._handle_interactions(space_held)

        # Calculate rewards
        reward = self._calculate_reward(prev_player_pos) + reward_from_events

        # Update state
        self.steps += 1
        terminated = self._check_termination()

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_movement(self, movement, shift_held):
        speed = self.PLAYER_SPEED * (self.SPRINT_MULTIPLIER if shift_held else 1)
        move_vec = pygame.Vector2(0, 0)
        
        if movement == 1: move_vec.y = -1 # Up
        elif movement == 2: move_vec.y = 1 # Down
        elif movement == 3: move_vec.x = -1 # Left
        elif movement == 4: move_vec.x = 1 # Right
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            new_pos = self.player_pos + move_vec * speed

            # Collision detection
            player_rect = pygame.Rect(0, 0, 12, 12)
            player_rect.center = new_pos
            
            if player_rect.left < 0 or player_rect.right > self.WIDTH or \
               player_rect.top < 0 or player_rect.bottom > self.HEIGHT:
                return # Boundary collision

            collides_with_wall = False
            for wall in self.walls:
                if wall.colliderect(player_rect):
                    collides_with_wall = True
                    break
            
            if not collides_with_wall:
                self.player_pos = new_pos

    def _update_ghosts(self):
        for ghost in self.ghosts:
            if not ghost["waypoints"]: continue
            target = ghost["waypoints"][ghost["waypoint_idx"]]
            direction = target - ghost["pos"]

            if direction.length() < self.GHOST_SPEED * 2:
                ghost["waypoint_idx"] = (ghost["waypoint_idx"] + 1) % len(ghost["waypoints"])
            else:
                direction.normalize_ip()
                ghost["pos"] += direction * self.GHOST_SPEED

    def _handle_interactions(self, space_held):
        event_reward = 0
        player_rect = pygame.Rect(0, 0, 12, 12)
        player_rect.center = self.player_pos

        # Item collection
        if space_held:
            for item in self.items[:]:
                if player_rect.colliderect(item):
                    self.items.remove(item)
                    self.score += 10
                    event_reward += 10
                    # Spawn collection particles
                    for _ in range(15):
                        self.particles.append(self._create_particle(item.center, self.COLOR_ITEM))
                    # sfx: item_collect.wav

        # Ghost collision
        for ghost in self.ghosts:
            ghost_rect = pygame.Rect(0, 0, 16, 16)
            ghost_rect.center = ghost["pos"]
            if player_rect.colliderect(ghost_rect):
                self.ghost_catches += 1
                event_reward -= 10
                # sfx: ghost_catch.wav
                # Spawn catch particles
                for _ in range(25):
                    self.particles.append(self._create_particle(self.player_pos, self.COLOR_GHOST))
                
                # Reset player to start, ghost to a random waypoint
                start_rect = self.floors[self.np_random.integers(len(self.floors))]
                self.player_pos = pygame.Vector2(start_rect.center)
                if ghost['waypoints']:
                    ghost['pos'] = pygame.Vector2(self.np_random.choice(ghost['waypoints']))
                break # Only one catch per frame

        # Exit collision
        if player_rect.colliderect(self.exit_rect):
            self.game_over = True
            self.game_won = True
            self.score += 100
            event_reward += 100
            # sfx: level_win.wav

        return event_reward

    def _calculate_reward(self, prev_player_pos):
        reward = 0
        
        # Reward for moving towards exit
        prev_dist_exit = prev_player_pos.distance_to(self.exit_rect.center)
        curr_dist_exit = self.player_pos.distance_to(self.exit_rect.center)
        if curr_dist_exit < prev_dist_exit:
            reward += 0.1
        
        # Penalty for moving towards nearest ghost
        if self.ghosts:
            prev_dists_ghosts = [prev_player_pos.distance_to(g["pos"]) for g in self.ghosts]
            curr_dists_ghosts = [self.player_pos.distance_to(g["pos"]) for g in self.ghosts]
            min_curr_dist = min(curr_dists_ghosts) if curr_dists_ghosts else float('inf')
            min_prev_dist = min(prev_dists_ghosts) if prev_dists_ghosts else float('inf')
            if min_curr_dist < min_prev_dist:
                reward -= 0.1

        return reward

    def _check_termination(self):
        if self.game_won:
            return True
        if self.ghost_catches >= self.GHOST_CATCH_LIMIT:
            self.game_over = True
            self.score -= 50 # Final penalty for losing
            # sfx: game_over.wav
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_particles()
        self.screen.blit(self._vignette_surface, (0, 0))
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Render floors and walls
        for floor_rect in self.floors:
            pygame.draw.rect(self.screen, self.COLOR_FLOOR, floor_rect)
        # for wall_rect in self.walls:
        #     pygame.draw.rect(self.screen, self.COLOR_WALL, wall_rect)

        # Render exit
        if self.exit_rect:
            pygame.draw.rect(self.screen, self.COLOR_EXIT, self.exit_rect)

        # Render items
        for item_rect in self.items:
            pygame.draw.rect(self.screen, self.COLOR_ITEM, item_rect)

        # Render ghosts
        for ghost in self.ghosts:
            bob = math.sin(self.steps * 0.1 + ghost["bob_offset"]) * 3
            pos = (int(ghost["pos"].x), int(ghost["pos"].y + bob))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 12, self.COLOR_GHOST_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 12, self.COLOR_GHOST_GLOW)
            # Body
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, self.COLOR_GHOST)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], 8, self.COLOR_GHOST)

        # Render player
        if self.player_pos:
            player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
            # Glow effect
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], 10, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], 10, self.COLOR_PLAYER_GLOW)
            # Body
            pygame.draw.rect(self.screen, self.COLOR_PLAYER, (player_pos_int[0]-6, player_pos_int[1]-6, 12, 12))

    def _create_particle(self, pos, color):
        return {
            "pos": pygame.Vector2(pos),
            "vel": pygame.Vector2(self.np_random.uniform(-2, 2), self.np_random.uniform(-2, 2)),
            "life": self.np_random.integers(20, 40),
            "color": color
        }

    def _render_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)
                continue
            
            # Fade out effect
            alpha = int(255 * (p["life"] / 40))
            color_with_alpha = (*p["color"], alpha)
            
            # Create a temporary surface for the particle to handle alpha
            particle_surf = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.rect(particle_surf, color_with_alpha, (0, 0, 4, 4))
            self.screen.blit(particle_surf, (int(p["pos"].x-2), int(p["pos"].y-2)))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Ghost catches
        catches_text = self.font_ui.render(f"CAUGHT: {self.ghost_catches}/{self.GHOST_CATCH_LIMIT}", True, self.COLOR_TEXT)
        self.screen.blit(catches_text, (self.WIDTH - catches_text.get_width() - 10, 10))
        
        # Steps
        steps_text = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH // 2 - steps_text.get_width() // 2, self.HEIGHT - 30))

        # Game over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.game_won:
                msg = "YOU ESCAPED!"
                color = self.COLOR_EXIT
            else:
                msg = "GAME OVER"
                color = self.COLOR_GHOST
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "ghost_catches": self.ghost_catches,
            "items_remaining": len(self.items),
            "player_pos": (self.player_pos.x, self.player_pos.y) if self.player_pos else (0,0),
            "exit_pos": self.exit_rect.center if self.exit_rect else (0,0)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset(seed=1)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=1)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


# Example usage for human play
if __name__ == '__main__':
    # We need a display for human play
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=random.randint(0, 1_000_000))
    
    # Setup Pygame window for display
    pygame.display.set_caption(env.game_description)
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    running = True
    terminated = False
    
    # Main game loop
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            # Space and Shift
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final Score: {info['score']}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for human play

    env.close()