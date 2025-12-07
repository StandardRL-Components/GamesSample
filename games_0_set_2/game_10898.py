import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:14:25.598482
# Source Brief: brief_00898.md
# Brief Index: 898
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Dodge obstacles in a fast-paced, auto-scrolling cyberpunk world. Use clones and portals to navigate the treacherous path and rack up combo points."
    )
    user_guide = (
        "Controls: ←→ to move. Press space to create a clone (hazard). Press shift to place portal entry/exit points."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (0, 150, 255)
    COLOR_PLAYER_GLOW = (0, 200, 255, 50)
    COLOR_CLONE = (0, 150, 255)
    COLOR_PORTAL_A = (200, 0, 255)
    COLOR_PORTAL_B = (255, 150, 0)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_OBSTACLE_GLOW = (255, 50, 50, 100)
    COLOR_FINISH = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_COMBO_BAR = (0, 255, 150)

    # Game Parameters
    PLAYER_SIZE = 12
    PLAYER_SPEED = 8
    CLONE_LIFETIME = 3 * FPS # 3 seconds
    PORTAL_LIFETIME = 10 * FPS # Disappear if not used
    MAX_STEPS = 5000
    LEVEL_LENGTH = 15000 # in pixels
    INITIAL_SCROLL_SPEED = 2.0
    OBSTACLE_SIZE = 40

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.Font(pygame.font.get_default_font(), 18)
            self.font_combo = pygame.font.Font(pygame.font.get_default_font(), 28)
        except IOError:
            self.font_ui = pygame.font.SysFont("monospace", 18)
            self.font_combo = pygame.font.SysFont("monospace", 28)

        # --- Internal State ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_pos = None
        self.player_trail = []
        self.clones = []
        self.portals = []
        self.obstacles = []
        self.particles = []
        self.camera_y = 0
        self.scroll_speed = 0
        self.combo = 0
        self.max_combo_this_step = 0
        self.last_generated_y = 0
        self.obstacle_density = 0
        self.prev_space_held = False
        self.prev_shift_held = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50)
        self.player_trail = []
        
        self.clones = []
        self.portals = []
        self.obstacles = []
        self.particles = []

        self.camera_y = 0
        self.scroll_speed = self.INITIAL_SCROLL_SPEED
        self.obstacle_density = 0.05

        self.combo = 0
        self.max_combo_this_step = 0
        self.last_generated_y = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_level_chunk()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.1  # Survival reward

        # --- Handle Input and State Updates ---
        self._handle_input(action)
        self._update_world()
        self._update_player()
        self._update_clones()
        self._update_portals()
        self._update_particles()
        
        # --- Collision and Event Checks ---
        collision_reward, terminated = self._check_collisions()
        reward += collision_reward
        
        combo_reward = self._check_combo()
        reward += combo_reward

        # --- Termination Conditions ---
        if self.player_pos.y < self.camera_y: # Went off top of screen
             terminated = True
             reward -= 100
        
        if self.player_pos.y > self.camera_y + self.SCREEN_HEIGHT: # Fell off bottom
             terminated = True
             reward -= 100
             
        if self.camera_y > self.LEVEL_LENGTH:
            terminated = True
            reward += 100 # Reached finish line
            # Sound: Victory!

        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward
        
        # Update previous action states
        self.prev_space_held = (action[1] == 1)
        self.prev_shift_held = (action[2] == 1)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        
        # Movement
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = max(self.PLAYER_SIZE, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_SIZE))

        # Clone (Spacebar press)
        space_pressed = space_action == 1 and not self.prev_space_held
        if space_pressed:
            self._spawn_clone()

        # Portal (Shift press)
        shift_pressed = shift_action == 1 and not self.prev_shift_held
        if shift_pressed:
            self._place_portal()

    def _update_world(self):
        # Scroll camera
        self.camera_y += self.scroll_speed
        
        # Difficulty scaling
        if self.steps > 0 and self.steps % 500 == 0:
            self.scroll_speed += 0.05
            self.obstacle_density = min(0.25, self.obstacle_density + 0.01)

        # Procedural generation
        if self.camera_y > self.last_generated_y - self.SCREEN_HEIGHT:
            self._generate_level_chunk()
            
        # Update player trail
        if self.steps % 2 == 0:
            self.player_trail.append({'pos': self.player_pos.copy(), 'life': 15})
        self.player_trail = [t for t in self.player_trail if t['life'] > 0]
        for t in self.player_trail:
            t['life'] -= 1

    def _update_player(self):
        # Player is also affected by scroll
        self.player_pos.y -= self.scroll_speed

    def _update_clones(self):
        for clone in self.clones:
            clone['lifetime'] -= 1
        self.clones = [c for c in self.clones if c['lifetime'] > 0]

    def _update_portals(self):
        for portal in self.portals:
            portal['lifetime'] -= 1
        self.portals = [p for p in self.portals if p['lifetime'] > 0]

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifetime'] -= 1
        self.particles = [p for p in self.particles if p['lifetime'] > 0]

    def _spawn_clone(self):
        # Sound: Clone created
        self.clones.append({
            'pos': self.player_pos.copy(),
            'lifetime': self.CLONE_LIFETIME
        })
        self._spawn_particles(self.player_pos, self.COLOR_CLONE, 20)

    def _place_portal(self):
        # Sound: Portal placed
        if len(self.portals) == 0:
            self.portals.append({
                'pos': self.player_pos.copy(),
                'type': 'A',
                'lifetime': self.PORTAL_LIFETIME
            })
            self._spawn_particles(self.player_pos, self.COLOR_PORTAL_A, 15)
        elif len(self.portals) == 1 and self.portals[0]['type'] == 'A':
            self.portals.append({
                'pos': self.player_pos.copy(),
                'type': 'B',
                'lifetime': self.PORTAL_LIFETIME
            })
            self._spawn_particles(self.player_pos, self.COLOR_PORTAL_B, 15)

    def _generate_level_chunk(self):
        grid_y_start = self.last_generated_y
        grid_y_end = grid_y_start - self.SCREEN_HEIGHT * 2
        
        num_cols = self.SCREEN_WIDTH // self.OBSTACLE_SIZE
        
        for y in range(grid_y_start, grid_y_end, -self.OBSTACLE_SIZE):
            gap_start = self.np_random.integers(0, num_cols - 3)
            gap_end = gap_start + 3 # Ensure a path of 3 blocks wide
            
            for x_idx in range(num_cols):
                if not (gap_start <= x_idx <= gap_end):
                    if self.np_random.random() < self.obstacle_density:
                        obs_x = x_idx * self.OBSTACLE_SIZE
                        obs_y = y
                        self.obstacles.append(pygame.Rect(obs_x, obs_y, self.OBSTACLE_SIZE, self.OBSTACLE_SIZE))
        
        self.last_generated_y = grid_y_end
        # Prune obstacles that are far below the camera
        self.obstacles = [o for o in self.obstacles if o.top > self.camera_y - self.SCREEN_HEIGHT]

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE, self.player_pos.y - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
        
        # Player vs Obstacles
        for obs in self.obstacles:
            if player_rect.colliderect(obs):
                # Sound: Collision/Explosion
                self._spawn_particles(self.player_pos, self.COLOR_OBSTACLE, 50)
                return -100, True

        # Player vs Clones
        for clone in self.clones:
            clone_rect = pygame.Rect(clone['pos'].x - self.PLAYER_SIZE, clone['pos'].y - self.PLAYER_SIZE, self.PLAYER_SIZE*2, self.PLAYER_SIZE*2)
            if player_rect.colliderect(clone_rect):
                # Sound: Collision/Explosion
                self._spawn_particles(self.player_pos, self.COLOR_CLONE, 50)
                return -100, True

        # Player vs Portals
        if len(self.portals) == 2:
            portal_a = self.portals[0]
            portal_b = self.portals[1]
            portal_a_rect = pygame.Rect(portal_a['pos'].x - 15, portal_a['pos'].y - 15, 30, 30)
            portal_b_rect = pygame.Rect(portal_b['pos'].x - 15, portal_b['pos'].y - 15, 30, 30)
            
            teleported = False
            if player_rect.colliderect(portal_a_rect):
                self.player_pos = portal_b['pos'].copy()
                teleported = True
            elif player_rect.colliderect(portal_b_rect):
                self.player_pos = portal_a['pos'].copy()
                teleported = True

            if teleported:
                # Sound: Teleport
                self._spawn_particles(self.player_pos, self.COLOR_PORTAL_A, 30)
                self._spawn_particles(self.player_pos, self.COLOR_PORTAL_B, 30)
                self.portals = [] # Portals disappear after use
                return 5, False

        return 0, False

    def _check_combo(self):
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_SIZE, self.player_pos.y - self.PLAYER_SIZE, self.PLAYER_SIZE * 2, self.PLAYER_SIZE * 2)
        combo_zone = player_rect.inflate(80, 80)
        
        nearby_hazards = 0
        all_hazards = self.obstacles + [c['pos'] for c in self.clones]
        
        for hazard in all_hazards:
            if isinstance(hazard, pygame.Rect):
                if combo_zone.colliderect(hazard):
                    nearby_hazards += 1
            else: # It's a clone position (Vector2)
                if combo_zone.collidepoint(hazard):
                    nearby_hazards += 1
        
        if nearby_hazards > self.max_combo_this_step:
            # Sound: Combo increase
            self.combo += (nearby_hazards - self.max_combo_this_step)
            reward = 1.0 * (nearby_hazards - self.max_combo_this_step)
            self.max_combo_this_step = nearby_hazards
            return reward
        
        if nearby_hazards == 0:
            self.max_combo_this_step = 0
            
        return 0

    def _spawn_particles(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'lifetime': random.randint(10, 20),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "combo": self.combo,
            "scroll_speed": self.scroll_speed
        }

    def _render_background(self):
        # Vertical grid lines
        for x in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        # Horizontal grid lines
        offset_y = self.camera_y % 40
        for y in range(int(-offset_y), self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _render_game(self):
        # Finish Line
        finish_y_screen = self.LEVEL_LENGTH - self.camera_y
        if 0 < finish_y_screen < self.SCREEN_HEIGHT:
            pygame.draw.line(self.screen, self.COLOR_FINISH, (0, finish_y_screen), (self.SCREEN_WIDTH, finish_y_screen), 5)

        # Obstacles
        for obs in self.obstacles:
            screen_rect = obs.move(0, -self.camera_y)
            if self.screen.get_rect().colliderect(screen_rect):
                pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, screen_rect, border_radius=3)
                pygame.gfxdraw.rectangle(self.screen, screen_rect, (*self.COLOR_OBSTACLE_GLOW[:3], 50))

        # Portals
        for i, portal in enumerate(self.portals):
            pos = portal['pos'] - pygame.Vector2(0, self.camera_y)
            color = self.COLOR_PORTAL_A if portal['type'] == 'A' else self.COLOR_PORTAL_B
            radius = 15 + 3 * math.sin(self.steps * 0.2 + i)
            self._draw_glow_circle(self.screen, color, (int(pos.x), int(pos.y)), int(radius), 100)

        # Clones
        for clone in self.clones:
            pos = clone['pos'] - pygame.Vector2(0, self.camera_y)
            alpha = int(255 * (clone['lifetime'] / self.CLONE_LIFETIME))
            self._draw_glow_circle(self.screen, self.COLOR_CLONE, (int(pos.x), int(pos.y)), self.PLAYER_SIZE, alpha)

        # Player Trail
        for t in self.player_trail:
            pos = t['pos'] - pygame.Vector2(0, self.camera_y)
            alpha = int(255 * (t['life'] / 15.0))
            color = (*self.COLOR_PLAYER, alpha)
            pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), self.PLAYER_SIZE * (t['life']/15.0))

        # Player
        if not self.game_over:
            player_screen_pos = self.player_pos - pygame.Vector2(0, self.camera_y)
            self._draw_glow_circle(self.screen, self.COLOR_PLAYER, (int(player_screen_pos.x), int(player_screen_pos.y)), self.PLAYER_SIZE, 255)

        # Particles
        for p in self.particles:
            pos = p['pos'] - pygame.Vector2(0, self.camera_y)
            alpha = int(255 * (p['lifetime'] / 20.0))
            color = (*p['color'], alpha)
            size = max(1, int(3 * (p['lifetime'] / 20.0)))
            pygame.draw.circle(self.screen, color, (int(pos.x), int(pos.y)), size)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Steps
        steps_surf = self.font_ui.render(f"STEPS: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 10, 10))
        
        # Combo
        combo_text_surf = self.font_combo.render("COMBO", True, self.COLOR_TEXT)
        self.screen.blit(combo_text_surf, (self.SCREEN_WIDTH // 2 - combo_text_surf.get_width() // 2, 10))
        combo_num_surf = self.font_combo.render(f"{self.combo}", True, self.COLOR_COMBO_BAR)
        self.screen.blit(combo_num_surf, (self.SCREEN_WIDTH // 2 - combo_num_surf.get_width() // 2, 40))
        
        # Progress Bar
        progress = self.camera_y / self.LEVEL_LENGTH
        bar_width = self.SCREEN_WIDTH - 20
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.SCREEN_HEIGHT - 20, bar_width, 10), 1)
        pygame.draw.rect(self.screen, self.COLOR_FINISH, (10, self.SCREEN_HEIGHT - 20, bar_width * progress, 10))

    def _draw_glow_circle(self, surface, color, center, radius, alpha):
        # Base solid circle
        pygame.gfxdraw.filled_circle(surface, center[0], center[1], radius, color)
        pygame.gfxdraw.aacircle(surface, center[0], center[1], radius, color)
        
        # Glow effect
        for i in range(4):
            glow_alpha = int(alpha * (1 - i/5) * 0.4)
            glow_radius = radius + i * 3
            if glow_radius > 0:
                pygame.gfxdraw.aacircle(surface, center[0], center[1], glow_radius, (*color[:3], glow_alpha))

    def close(self):
        pygame.quit()

# Example of how to run the environment
if __name__ == '__main__':
    # The validation function is removed as it's not part of the standard API
    # and was just for development. The main block is for human play.
    env = GameEnv()
    
    # --- Manual Play ---
    # Use Arrow Keys for Left/Right, Space to Clone, Left Shift to Portal
    
    obs, info = env.reset()
    done = False
    
    # Re-initialize pygame for display if not already done for rendering
    if "human" in env.metadata["render_modes"]:
        display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Cyberpunk Runner")
    
    clock = pygame.time.Clock()
    
    while not done:
        # Action mapping for human player
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display window
        # This requires a display to be initialized.
        # We need to set up a display for human play.
        if 'display_screen' not in locals():
            pygame.display.init()
            display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
            pygame.display.set_caption("Cyberpunk Runner")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(env.FPS)
        
    print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    env.close()
    pygame.quit()