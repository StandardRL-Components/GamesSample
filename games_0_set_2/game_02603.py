
# Generated: 2025-08-28T05:23:08.473484
# Source Brief: brief_02603.md
# Brief Index: 2603

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Collect all 7 souls before time runs out. Avoid the ghosts!"
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Collect wandering souls in a haunted graveyard while evading patrolling ghosts to complete your shift."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_WALL = (50, 60, 70)
    COLOR_OBSTACLE = (40, 50, 60)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_SOUL = (200, 255, 255)
    COLOR_SOUL_GLOW = (150, 200, 200)
    COLOR_GHOST = (200, 200, 220)
    COLOR_TEXT = (220, 220, 230)
    COLOR_WIN = (100, 255, 100)
    COLOR_LOSE = (255, 100, 100)

    # Game parameters
    PLAYER_RADIUS = 10
    PLAYER_SPEED = 4.0
    SOUL_RADIUS = 6
    GHOST_RADIUS = 12
    GHOST_SPEED = 1.5
    NUM_SOULS = 7
    NUM_OBSTACLES = 10
    INITIAL_HEALTH = 2
    INVINCIBILITY_FRAMES = 60 # 2 seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables are initialized in reset()
        self.player_pos = None
        self.souls = None
        self.ghosts = None
        self.obstacles = None
        self.particles = None
        self.steps = None
        self.score = None
        self.health = None
        self.souls_collected = None
        self.time_left_steps = None
        self.invincibility_timer = None
        self.game_over = None

        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.health = self.INITIAL_HEALTH
        self.souls_collected = 0
        self.time_left_steps = self.GAME_DURATION_SECONDS * self.FPS
        self.invincibility_timer = 0
        self.game_over = False
        self.particles = []

        self._generate_layout()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_layout(self):
        # Generate a safe spawn area in the center
        safe_zone = pygame.Rect(self.SCREEN_WIDTH * 0.4, self.SCREEN_HEIGHT * 0.4,
                                self.SCREEN_WIDTH * 0.2, self.SCREEN_HEIGHT * 0.2)
        
        self.player_pos = pygame.math.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        # Generate obstacles (gravestones)
        self.obstacles = []
        for _ in range(self.NUM_OBSTACLES):
            while True:
                w = self.np_random.integers(20, 60)
                h = self.np_random.integers(20, 60)
                x = self.np_random.uniform(20, self.SCREEN_WIDTH - w - 20)
                y = self.np_random.uniform(20, self.SCREEN_HEIGHT - h - 20)
                new_obstacle = pygame.Rect(x, y, w, h)
                if not new_obstacle.colliderect(safe_zone):
                    self.obstacles.append(new_obstacle)
                    break
        
        # Generate souls
        self.souls = []
        for _ in range(self.NUM_SOULS):
            while True:
                x = self.np_random.uniform(20, self.SCREEN_WIDTH - 20)
                y = self.np_random.uniform(20, self.SCREEN_HEIGHT - 20)
                new_soul_pos = pygame.math.Vector2(x, y)
                
                valid_pos = True
                if new_soul_pos.distance_to(self.player_pos) < 50:
                    valid_pos = False
                if valid_pos:
                    for obs in self.obstacles:
                        if obs.collidepoint(new_soul_pos):
                            valid_pos = False; break
                if valid_pos:
                    for soul_pos in self.souls:
                        if soul_pos.distance_to(new_soul_pos) < self.SOUL_RADIUS * 4:
                            valid_pos = False; break
                
                if valid_pos:
                    self.souls.append(new_soul_pos)
                    break

        # Generate ghosts with patrol paths
        self.ghosts = []
        path1 = [pygame.math.Vector2(p) for p in [(100, 100), (self.SCREEN_WIDTH - 100, 100), (self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 100), (100, self.SCREEN_HEIGHT - 100)]]
        path2 = [pygame.math.Vector2(p) for p in [(self.SCREEN_WIDTH / 2, 50), (self.SCREEN_WIDTH - 50, self.SCREEN_HEIGHT / 2), (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50), (50, self.SCREEN_HEIGHT / 2)]]
        self.ghosts.append({'pos': pygame.math.Vector2(path1[0]), 'path': path1, 'target_idx': 1})
        self.ghosts.append({'pos': pygame.math.Vector2(path2[0]), 'path': path2, 'target_idx': 1})
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0.1  # Survival reward

        # --- Update Game Logic ---
        self._handle_input(action)
        self._update_ghosts()
        self._update_particles()
        
        if self.invincibility_timer > 0:
            self.invincibility_timer -= 1
        
        reward += self._check_soul_collisions()
        reward += self._check_ghost_collisions()
        
        self.steps += 1
        self.time_left_steps -= 1
        
        # --- Check Termination ---
        terminated = False
        if self.souls_collected >= self.NUM_SOULS:
            reward += 100
            self.score += 100
            terminated = True
            self.game_over = "win"
        elif self.health <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = "lose_ghost"
        elif self.time_left_steps <= 0:
            reward -= 100
            self.score -= 100
            terminated = True
            self.game_over = "lose_time"
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, action):
        movement = action[0]
        
        move_vec = pygame.math.Vector2(0, 0)
        if movement == 1: move_vec.y = -1
        elif movement == 2: move_vec.y = 1
        elif movement == 3: move_vec.x = -1
        elif movement == 4: move_vec.x = 1
        
        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self._move_player(move_vec * self.PLAYER_SPEED)

    def _move_player(self, velocity):
        new_pos = self.player_pos + velocity
        
        player_rect_x = pygame.Rect(new_pos.x - self.PLAYER_RADIUS, self.player_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)
        player_rect_y = pygame.Rect(self.player_pos.x - self.PLAYER_RADIUS, new_pos.y - self.PLAYER_RADIUS, self.PLAYER_RADIUS * 2, self.PLAYER_RADIUS * 2)

        collided_x = False
        collided_y = False

        for obs in self.obstacles:
            if obs.colliderect(player_rect_x):
                collided_x = True
            if obs.colliderect(player_rect_y):
                collided_y = True
        
        if not collided_x:
            self.player_pos.x = new_pos.x
        if not collided_y:
            self.player_pos.y = new_pos.y

        self.player_pos.x = np.clip(self.player_pos.x, self.PLAYER_RADIUS, self.SCREEN_WIDTH - self.PLAYER_RADIUS)
        self.player_pos.y = np.clip(self.player_pos.y, self.PLAYER_RADIUS, self.SCREEN_HEIGHT - self.PLAYER_RADIUS)

    def _update_ghosts(self):
        for ghost in self.ghosts:
            target_pos = ghost['path'][ghost['target_idx']]
            if ghost['pos'].distance_to(target_pos) < self.GHOST_SPEED:
                ghost['target_idx'] = (ghost['target_idx'] + 1) % len(ghost['path'])
            else:
                direction = (target_pos - ghost['pos']).normalize()
                ghost['pos'] += direction * self.GHOST_SPEED

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['life'] -= 1
            p['radius'] = max(0, p['radius'] * 0.95)

    def _check_soul_collisions(self):
        reward = 0
        collected_souls_indices = []
        for i, soul_pos in enumerate(self.souls):
            if self.player_pos.distance_to(soul_pos) < self.PLAYER_RADIUS + self.SOUL_RADIUS:
                collected_souls_indices.append(i)
                self.souls_collected += 1
                reward += 10
                self.score += 10
                self._create_soul_particles(soul_pos)
                # sfx: soul_collect
        
        if collected_souls_indices:
            self.souls = [soul for i, soul in enumerate(self.souls) if i not in collected_souls_indices]
        return reward

    def _check_ghost_collisions(self):
        if self.invincibility_timer > 0:
            return 0
        
        for ghost in self.ghosts:
            if self.player_pos.distance_to(ghost['pos']) < self.PLAYER_RADIUS + self.GHOST_RADIUS:
                self.health -= 1
                self.invincibility_timer = self.INVINCIBILITY_FRAMES
                # sfx: player_hit
                return -5
        return 0

    def _create_soul_particles(self, pos):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                'pos': pos.copy(),
                'vel': pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed,
                'life': self.np_random.integers(15, 30),
                'radius': self.np_random.uniform(2, 5),
                'color': self.COLOR_SOUL
            })
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Boundary wall and obstacles
        pygame.draw.rect(self.screen, self.COLOR_WALL, (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 5)
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obs)

        # Particles
        for p in self.particles:
            pos = (int(p['pos'].x), int(p['pos'].y))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], int(p['radius']), p['color'])

        # Souls
        for soul_pos in self.souls:
            pos = (int(soul_pos.x), int(soul_pos.y))
            glow_radius = int(self.SOUL_RADIUS * 1.8 + math.sin(self.steps * 0.2) * 2)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_SOUL_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], glow_radius, self.COLOR_SOUL_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.SOUL_RADIUS, self.COLOR_SOUL)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.SOUL_RADIUS, self.COLOR_SOUL)

        # Ghosts
        ghost_surface = pygame.Surface((self.GHOST_RADIUS * 2, self.GHOST_RADIUS * 2), pygame.SRCALPHA)
        if self.steps % 10 < 8: # Flicker effect
            alpha = 100 + math.sin(self.steps * 0.3) * 20
            pygame.gfxdraw.filled_circle(ghost_surface, self.GHOST_RADIUS, self.GHOST_RADIUS, self.GHOST_RADIUS, (*self.COLOR_GHOST, alpha))
            pygame.gfxdraw.aacircle(ghost_surface, self.GHOST_RADIUS, self.GHOST_RADIUS, self.GHOST_RADIUS, (*self.COLOR_GHOST, alpha))
            for ghost in self.ghosts:
                self.screen.blit(ghost_surface, (int(ghost['pos'].x - self.GHOST_RADIUS), int(ghost['pos'].y - self.GHOST_RADIUS)))

        # Player
        player_pos_int = (int(self.player_pos.x), int(self.player_pos.y))
        if self.health >= 2: pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS + 8, self.COLOR_PLAYER)
        if self.health >= 1: pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS + 4, self.COLOR_PLAYER)
        
        is_invincible_flicker = self.invincibility_timer > 0 and self.steps % 10 < 5
        if not is_invincible_flicker:
            pygame.gfxdraw.filled_circle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, player_pos_int[0], player_pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

    def _render_ui(self):
        score_text = self.font_ui.render(f"Souls: {self.souls_collected}/{self.NUM_SOULS}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))
        
        time_left_sec = max(0, self.time_left_steps // self.FPS)
        timer_text = self.font_ui.render(f"Time: {time_left_sec}", True, self.COLOR_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 15, 10))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg, color = ("SHIFT COMPLETE", self.COLOR_WIN) if self.game_over == "win" else ("GAME OVER", self.COLOR_LOSE)
            text = self.font_game_over.render(msg, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.health,
            "souls_collected": self.souls_collected,
        }

    def close(self):
        pygame.quit()

# Example usage for human play
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Graveyard Shift")
    clock = pygame.time.Clock()
    
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    print("\n" + "="*30)
    print("      Graveyard Shift")
    print("="*30)
    print(env.game_description)
    print(env.user_guide)
    print("Press 'R' to reset the game.\n")
    
    while running:
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
        
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()
    pygame.quit()