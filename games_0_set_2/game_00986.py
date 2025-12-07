# Generated: 2025-08-27T15:25:22.448892
# Source Brief: brief_00986.md
# Brief Index: 986

        
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
    """
    A side-scrolling platformer where a robot must navigate procedurally generated
    levels, avoid obstacles, and reach checkpoints within a time limit.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: ←→ to move, ↑ or Space to jump. Reach the green flag to advance."
    )
    game_description = (
        "Guide a jumping robot through a perilous, procedurally generated world. "
        "Avoid falling debris and treacherous pits to reach the goal before time runs out."
    )

    # Frame advance behavior
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.FPS = 50
        self.MAX_STEPS = 3000

        # --- Physics Constants ---
        self.PLAYER_SPEED = 4.0
        self.GRAVITY = 0.4
        self.JUMP_STRENGTH = -10.0
        self.PLAYER_WIDTH = 24
        self.PLAYER_HEIGHT = 32

        # --- Color Palette ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_FAR = (30, 35, 50)
        self.COLOR_GRID_NEAR = (40, 45, 65)
        self.COLOR_PLAYER = (60, 160, 255)
        self.COLOR_PLAYER_EYE = (255, 255, 255)
        self.COLOR_PLATFORM = (120, 130, 150)
        self.COLOR_PIT = (180, 50, 50)
        self.COLOR_DEBRIS = (140, 90, 40)
        self.COLOR_CHECKPOINT = (80, 220, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        
        self.player_world_x = 0
        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.player_jump_squash = 0

        self.world_offset_x = 0
        self.platforms = []
        self.debris = []
        self.particles = []
        self.checkpoint_pos = pygame.Vector2(0, 0)
        
        self.current_stage = 1
        self.time_left = 0
        self.debris_spawn_timer = 0
        self.debris_spawn_rate = 0
        
        self.last_reward_info = {}
        self.takeoff_platform_rect = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize RNG
        if self.np_random is None or seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset core state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        
        # Reset player
        self.player_world_x = 100
        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH // 3, 100)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_on_ground = False
        self.player_jump_squash = 0

        # Reset world
        self.world_offset_x = 0
        self.particles = []
        self.debris = []
        self._setup_stage()

        return self._get_observation(), self._get_info()

    def _setup_stage(self):
        """Initializes the game state for the current stage."""
        self.time_left = 60 * self.FPS
        self.debris_spawn_rate = 0.02 + (self.current_stage - 1) * 0.01
        self.debris_spawn_timer = self.np_random.integers(0, self.FPS)
        self.platforms = []
        self._generate_stage()
        
        # Reset player to start of stage
        start_platform = self.platforms[0]
        self.player_world_x = start_platform['rect'].centerx
        self.player_pos.y = start_platform['rect'].top - self.PLAYER_HEIGHT
        self.player_vel.x = 0
        self.player_vel.y = 0
        self.takeoff_platform_rect = None

    def _generate_stage(self):
        """Procedurally generates platforms for the current stage."""
        plat_x = 0
        plat_y = self.SCREEN_HEIGHT - 50
        
        # Difficulty parameters
        min_w = 200 - self.current_stage * 30
        max_w = 400 - self.current_stage * 60
        min_gap = 50 + self.current_stage * 10
        max_gap = 100 + self.current_stage * 15
        max_dy = 40 + self.current_stage * 20
        move_chance = 0.1 * self.current_stage

        # Starting platform
        self.platforms.append({'rect': pygame.Rect(plat_x, plat_y, 300, 50), 'type': 'static'})

        for _ in range(20): # Generate 20 segments
            last_plat = self.platforms[-1]['rect']
            plat_x = last_plat.right + self.np_random.integers(min_gap, max_gap)
            plat_y = np.clip(
                last_plat.y + self.np_random.integers(-max_dy, max_dy),
                self.SCREEN_HEIGHT // 2,
                self.SCREEN_HEIGHT - 30
            )
            width = self.np_random.integers(min_w, max_w)
            
            plat_type = 'static'
            move_speed = 0
            if self.np_random.random() < move_chance:
                plat_type = 'moving'
                move_speed = (0.5 + self.np_random.random() * self.current_stage) * self.np_random.choice([-1, 1])

            self.platforms.append({
                'rect': pygame.Rect(plat_x, plat_y, width, 50), 
                'type': plat_type, 
                'speed': move_speed,
                'center': plat_x + width / 2,
                'range': 100
            })
        
        # Checkpoint
        end_platform = self.platforms[-1]['rect']
        self.checkpoint_pos = pygame.Vector2(end_platform.centerx, end_platform.top)

    def step(self, action):
        if self.game_over:
            terminated = self._check_termination()
            truncated = self.steps >= self.MAX_STEPS
            return self._get_observation(), 0, terminated, truncated, self._get_info()

        # Unpack action
        movement, space_held, _ = action
        
        reward = self._update_game_state(movement, space_held)
        
        self.steps += 1
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        bonus = 0
        if terminated:
            self.game_over = True
            if self.current_stage > 3: # Win condition
                bonus = 100
            else: # Loss condition
                bonus = -100
        
        if truncated:
            self.game_over = True

        self.score += reward + bonus

        return (
            self._get_observation(),
            reward + bonus,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_state(self, movement, space_held):
        """Main logic update loop."""
        reward = 0.1 # Survival reward

        # --- Player Input ---
        is_jumping = movement == 1 or space_held
        if is_jumping and self.player_on_ground:
            self.player_vel.y = self.JUMP_STRENGTH
            self.player_on_ground = False
            self.player_jump_squash = 10
            self._create_particles(self.player_pos.x, self.player_pos.y + self.PLAYER_HEIGHT, 10, self.COLOR_PLATFORM)
            if self.last_reward_info.get('on_platform'):
                self.takeoff_platform_rect = self.last_reward_info['platform_rect']

        if movement == 3: # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # --- Physics & Collisions ---
        # Apply gravity
        if not self.player_on_ground:
            self.player_vel.y += self.GRAVITY
            self.player_vel.y = min(self.player_vel.y, 15) # Terminal velocity

        # Update world position
        self.player_world_x += self.player_vel.x
        self.player_pos.y += self.player_vel.y
        
        # Update moving platforms
        for plat in self.platforms:
            if plat['type'] == 'moving':
                plat['rect'].x += plat['speed']
                if abs(plat['rect'].centerx - plat['center']) > plat['range']:
                    plat['speed'] *= -1

        # Collision detection with platforms
        self.player_on_ground = False
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_WIDTH / 2, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        
        current_platform_rect = None
        for plat in self.platforms:
            plat_screen_rect = plat['rect'].copy()
            plat_screen_rect.x += self.world_offset_x
            
            if player_rect.colliderect(plat_screen_rect):
                # Check if player was above the platform in the previous frame
                prev_player_bottom = player_rect.bottom - self.player_vel.y
                # FIX: Use >= 0 to handle being stationary on a platform, not just landing.
                if self.player_vel.y >= 0 and prev_player_bottom <= plat_screen_rect.top + 1:
                    self.player_pos.y = plat_screen_rect.top - self.PLAYER_HEIGHT
                    self.player_vel.y = 0
                    self.player_on_ground = True
                    self.player_jump_squash = -10
                    current_platform_rect = plat['rect']
                    break

        # --- Rewards based on state ---
        self.last_reward_info = {'on_platform': self.player_on_ground, 'platform_rect': current_platform_rect}
        if self.player_on_ground:
            # Risky jump reward
            if self.takeoff_platform_rect is not None and current_platform_rect is not None:
                gap = max(0, current_platform_rect.left - self.takeoff_platform_rect.right, self.takeoff_platform_rect.left - current_platform_rect.right)
                if gap > 2 * self.PLAYER_WIDTH:
                    reward += 5.0
                    self._create_particles(self.player_pos.x, self.player_pos.y, 20, (255, 215, 0)) # Gold particles
                self.takeoff_platform_rect = None
            
            # Penalty for camping on wide platforms
            if current_platform_rect and current_platform_rect.width > 300:
                reward -= 0.2
        
        # --- Update World State ---
        # Debris
        self.debris_spawn_timer -= 1
        if self.debris_spawn_timer <= 0:
            debris_x = self.player_world_x + self.np_random.uniform(-self.SCREEN_WIDTH, self.SCREEN_WIDTH)
            self.debris.append(pygame.Rect(debris_x, -20, 20, 20))
            self.debris_spawn_timer = self.np_random.uniform(0.5, 1.5) * self.FPS / self.debris_spawn_rate

        for d in self.debris[:]:
            d.y += 5 # Debris fall speed
            if d.top > self.SCREEN_HEIGHT:
                self.debris.remove(d)

        # Particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Particle gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
        
        # Timer
        self.time_left -= 1

        # Checkpoint
        checkpoint_dist = abs(self.player_world_x - self.checkpoint_pos.x)
        if checkpoint_dist < 30 and self.player_pos.y < self.checkpoint_pos.y:
            self.current_stage += 1
            reward += 10.0
            self._create_particles(self.player_pos.x, self.player_pos.y, 50, self.COLOR_CHECKPOINT)
            if self.current_stage <= 3:
                self._setup_stage()
        
        return reward

    def _check_termination(self):
        """Checks for game over conditions."""
        # Fell in a pit
        if self.player_pos.y > self.SCREEN_HEIGHT:
            return True
        
        # Hit by debris
        player_rect = pygame.Rect(self.player_pos.x - self.PLAYER_WIDTH / 2, self.player_pos.y, self.PLAYER_WIDTH, self.PLAYER_HEIGHT)
        for d in self.debris:
            debris_screen_rect = d.copy()
            debris_screen_rect.x += self.world_offset_x
            if player_rect.colliderect(debris_screen_rect):
                self._create_particles(self.player_pos.x, self.player_pos.y, 50, self.COLOR_PLAYER)
                return True
        
        # Out of time
        if self.time_left <= 0:
            return True
        
        # FIX: Removed MAX_STEPS check, as it's for truncation, not termination.
            
        # Completed all stages
        if self.current_stage > 3:
            return True

        return False

    def _get_observation(self):
        # Update camera offset
        self.world_offset_x = self.player_pos.x - self.player_world_x

        # Render all game elements
        self._render_background()
        self._render_world()
        self._render_player()
        self._render_particles()
        self._render_ui()

        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        # Parallax grids
        for i in range(0, self.SCREEN_WIDTH, 40):
            offset = int(self.world_offset_x * 0.1) % 40
            pygame.draw.line(self.screen, self.COLOR_GRID_FAR, (i + offset, 0), (i + offset, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID_FAR, (0, i), (self.SCREEN_WIDTH, i))
        
        for i in range(0, self.SCREEN_WIDTH, 80):
            offset = int(self.world_offset_x * 0.2) % 80
            pygame.draw.line(self.screen, self.COLOR_GRID_NEAR, (i + offset, 0), (i + offset, self.SCREEN_HEIGHT))
        for i in range(0, self.SCREEN_HEIGHT, 80):
            pygame.draw.line(self.screen, self.COLOR_GRID_NEAR, (0, i), (self.SCREEN_WIDTH, i))

    def _render_world(self):
        # Platforms
        for plat in self.platforms:
            screen_rect = plat['rect'].copy()
            screen_rect.x += self.world_offset_x
            pygame.draw.rect(self.screen, self.COLOR_PLATFORM, screen_rect, border_radius=3)
            # Pit indicator below platforms
            pygame.gfxdraw.box(self.screen, 
                               (screen_rect.left, screen_rect.bottom, screen_rect.width, 10), 
                               (*self.COLOR_PIT, 100))

        # Debris
        for d in self.debris:
            screen_rect = d.copy()
            screen_rect.x += self.world_offset_x
            pygame.draw.rect(self.screen, self.COLOR_DEBRIS, screen_rect)

        # Checkpoint
        cp_x = self.checkpoint_pos.x + self.world_offset_x
        cp_y = self.checkpoint_pos.y
        pygame.draw.line(self.screen, self.COLOR_CHECKPOINT, (cp_x, cp_y), (cp_x, cp_y - 40), 3)
        pygame.draw.polygon(self.screen, self.COLOR_CHECKPOINT, [(cp_x, cp_y - 40), (cp_x + 20, cp_y - 30), (cp_x, cp_y - 20)])

    def _render_player(self):
        if self.game_over and not (self.current_stage > 3): return

        # Squash and stretch effect
        if self.player_jump_squash != 0:
            squash_factor = abs(self.player_jump_squash) / 10.0
            if self.player_jump_squash > 0: # Stretching for jump
                w = self.PLAYER_WIDTH * (1 - 0.2 * squash_factor)
                h = self.PLAYER_HEIGHT * (1 + 0.4 * squash_factor)
            else: # Squashing for land
                w = self.PLAYER_WIDTH * (1 + 0.4 * squash_factor)
                h = self.PLAYER_HEIGHT * (1 - 0.3 * squash_factor)
            self.player_jump_squash -= 1 if self.player_jump_squash > 0 else -1
        else:
            w, h = self.PLAYER_WIDTH, self.PLAYER_HEIGHT

        player_rect = pygame.Rect(self.player_pos.x - w/2, self.player_pos.y + (self.PLAYER_HEIGHT - h), w, h)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        
        # Eye
        eye_x = player_rect.centerx + (5 if self.player_vel.x > 0 else -5 if self.player_vel.x < 0 else 0)
        eye_y = player_rect.centery - 5
        pygame.draw.circle(self.screen, self.COLOR_PLAYER_EYE, (int(eye_x), int(eye_y)), 4)

    def _render_particles(self):
        for p in self.particles:
            size = max(1, int(p['life'] / p['max_life'] * 5))
            pygame.draw.circle(self.screen, p['color'], p['pos'], size)

    def _render_ui(self):
        # Helper to draw text with shadow
        def draw_text(text, font, color, pos, anchor="topleft"):
            text_surf = font.render(text, True, color)
            shadow_surf = font.render(text, True, self.COLOR_TEXT_SHADOW)
            text_rect = text_surf.get_rect()
            setattr(text_rect, anchor, pos)
            self.screen.blit(shadow_surf, (text_rect.x + 2, text_rect.y + 2))
            self.screen.blit(text_surf, text_rect)

        # Score
        draw_text(f"SCORE: {int(self.score)}", self.font_medium, self.COLOR_TEXT, (10, 10))
        
        # Stage
        stage_text = f"WIN!" if self.current_stage > 3 else f"STAGE: {self.current_stage}"
        draw_text(stage_text, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH / 2, 10), anchor="midtop")

        # Timer
        time_str = f"TIME: {max(0, self.time_left / self.FPS):.1f}"
        draw_text(time_str, self.font_medium, self.COLOR_TEXT, (self.SCREEN_WIDTH - 10, 10), anchor="topright")
        
        if self.game_over:
            result_text = "WIN!" if self.current_stage > 3 else "GAME OVER"
            draw_text(result_text, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 30), anchor="center")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "time_left": self.time_left / self.FPS,
        }

    def _create_particles(self, x, y, count, color):
        for _ in range(count):
            life = self.np_random.integers(20, 40)
            self.particles.append({
                'pos': pygame.Vector2(x, y),
                'vel': pygame.Vector2(self.np_random.uniform(-3, 3), self.np_random.uniform(-4, 0)),
                'life': life,
                'max_life': life,
                'color': color,
            })

    def close(self):
        pygame.quit()


# Example of how to run the environment
if __name__ == '__main__':
    # To run with display, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    
    pygame.display.set_caption("Jumping Robot Rescue")
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # action = [movement, space, shift]
    # movement: 0=none, 1=up, 2=down, 3=left, 4=right
    action = [0, 0, 0] 

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        
        # Reset action
        action = [0, 0, 0]

        # Movement
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Space
        if keys[pygame.K_SPACE]:
            action[1] = 1
        
        # Shift (not used in this game)
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Display the rendered frame ---
        # Pygame uses (width, height) but numpy uses (height, width)
        # So we need to transpose the observation back
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

    env.close()