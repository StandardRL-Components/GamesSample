import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:13:14.513786
# Source Brief: brief_03017.md
# Brief Index: 3017
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    An Aztec-themed puzzle-platformer Gymnasium environment.
    The agent must navigate a crumbling temple and use terraforming tools
    to create a path for water to flow to the city center.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a crumbling Aztec temple, using terraforming tools to create a path for sacred water to flow to the city center."
    )
    user_guide = (
        "Use ←→ to move and ↑ to jump. Press Shift to switch tools and Space to use the selected tool."
    )
    auto_advance = True

    # --- CONSTANTS ---
    WIDTH, HEIGHT = 640, 400
    FPS = 30
    MAX_STEPS = 1500

    # Colors
    COLOR_BG_SKY = (40, 50, 80)
    COLOR_BG_MOUNTAINS = (60, 70, 100)
    COLOR_STONE = (120, 120, 120)
    COLOR_EARTH = (139, 69, 19)
    COLOR_PLAYER = (0, 220, 220)
    COLOR_PLAYER_GLOW = (0, 220, 220, 50)
    COLOR_WATER = (50, 150, 255, 180)
    COLOR_TARGET_GOLD = (255, 215, 0)
    COLOR_UI_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)
    COLOR_PROGRESS_EMPTY = (80, 80, 80)
    COLOR_PROGRESS_FILL = (50, 150, 255)
    
    # Physics
    GRAVITY = 0.8
    PLAYER_SPEED = 5.0
    JUMP_STRENGTH = -12.0
    MAX_FALL_SPEED = 15.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 24, bold=True)

        self.player_pos = pygame.Vector2(0, 0)
        self.player_vel = pygame.Vector2(0, 0)
        self.player_rect = pygame.Rect(0, 0, 20, 30)
        self.on_ground = False
        self.current_tool = 0  # 0: Earth, 1: Water
        self.tools = ["EARTH", "WATER"]
        
        self.platforms = []
        self.tool_targets = []
        self.water_basins = []
        
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.total_water_capacity = 1
        self.current_water_volume = 0
        self.last_dist_to_goal = 0

        # self.reset() is called by the wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._initialize_level()

        # Start player on the first platform to prevent falling immediately
        self.player_pos = pygame.Vector2(80, 350)
        self.player_vel = pygame.Vector2(0, 0)
        self.on_ground = True
        self.current_tool = 0
        
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self.last_dist_to_goal = self._get_dist_to_goal()

        return self._get_observation(), self._get_info()

    def _initialize_level(self):
        self.platforms = [
            # Starting platform
            {'rect': pygame.Rect(0, 350, 200, 50), 'type': 'stone', 'state': {}},
            # Gap platform
            {'rect': pygame.Rect(250, 350, 100, 50), 'type': 'stone', 'state': {}},
            # Crumbling platform
            {'rect': pygame.Rect(400, 300, 80, 20), 'type': 'crumble', 'state': {'health': 100}},
            # Final platform before goal
            {'rect': pygame.Rect(500, 250, 140, 150), 'type': 'stone', 'state': {}},
            # Hidden platform to be raised
            {'rect': pygame.Rect(250, 400, 100, 50), 'type': 'earth', 'state': {'target_y': 400, 'is_moving': False}},
        ]
        
        self.tool_targets = [
            # Target to raise the earth platform
            {'pos': pygame.Vector2(300, 340), 'type': 'earth', 'activated': False, 'radius': 15, 'linked_platform_idx': 4, 'linked_y': 350},
            # Target to activate water
            {'pos': pygame.Vector2(570, 240), 'type': 'water', 'activated': False, 'radius': 15, 'linked_basin_idx': 0},
        ]
        
        self.water_basins = [
            {'rect': pygame.Rect(520, 150, 100, 100), 'volume': 0, 'capacity': 100, 'is_filling': False}
        ]
        self.total_water_capacity = sum(b['capacity'] for b in self.water_basins)
        self.current_water_volume = 0


    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0.0
        self.game_over = False

        # --- 1. Handle Input and Tool Actions ---
        reward += self._handle_input(movement, space_held, shift_held)

        # --- 2. Update Player Physics ---
        self._update_player_physics()

        # --- 3. Update Game World ---
        self._update_world_state()

        # --- 4. Calculate Rewards ---
        dist_to_goal = self._get_dist_to_goal()
        if dist_to_goal < self.last_dist_to_goal:
            reward += 0.01
        elif dist_to_goal > self.last_dist_to_goal:
            reward -= 0.01
        self.last_dist_to_goal = dist_to_goal

        # --- 5. Check Termination Conditions ---
        if self.player_pos.y > self.HEIGHT + self.player_rect.height:
            self.game_over = True
            reward = -100
            self.score -= 100
        
        if self.current_water_volume >= self.total_water_capacity:
            self.game_over = True
            reward = 100
            self.score += 100

        self.steps += 1
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated and not terminated:
             reward -= 10 # Penalty for running out of time

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Horizontal Movement
        if movement == 3:  # Left
            self.player_vel.x = -self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_vel.x = self.PLAYER_SPEED
        else:
            self.player_vel.x = 0

        # Jumping
        if movement == 1 and self.on_ground:  # Up
            self.player_vel.y = self.JUMP_STRENGTH
            self.on_ground = False
            # sfx: PlayerJump.wav
            self._spawn_particles(self.player_rect.midbottom, 5, (200,200,200), (-1,1), (-1,-0.5))


        # Tool Switching (on press, not hold)
        if shift_held and not self.prev_shift_held:
            self.current_tool = (self.current_tool + 1) % len(self.tools)
            # sfx: ToolSwitch.wav

        # Use Tool (on press, not hold)
        if space_held and not self.prev_space_held:
            return self._use_tool()
            
        return 0

    def _use_tool(self):
        reward = -0.1 # Small penalty for using tool and failing
        tool_name = self.tools[self.current_tool]
        
        for target in self.tool_targets:
            if not target['activated'] and target['type'].lower() == tool_name.lower():
                dist = self.player_pos.distance_to(target['pos'])
                if dist < 50: # Activation range
                    target['activated'] = True
                    # sfx: ToolActivate.wav
                    self._spawn_particles(target['pos'], 20, self.COLOR_TARGET_GOLD, (-2,2), (-2,2), 30)

                    if target['type'] == 'earth':
                        # sfx: EarthRumble.wav
                        p_idx = target['linked_platform_idx']
                        self.platforms[p_idx]['state']['is_moving'] = True
                        self.platforms[p_idx]['state']['target_y'] = target['linked_y']
                        self.score += 10
                        return 1.0 # Reward for creating a platform

                    elif target['type'] == 'water':
                        # sfx: WaterGush.wav
                        b_idx = target['linked_basin_idx']
                        self.water_basins[b_idx]['is_filling'] = True
                        self.score += 5
                        return 5.0 # Reward for activating water

        return reward

    def _update_player_physics(self):
        # Apply gravity
        self.player_vel.y += self.GRAVITY
        self.player_vel.y = min(self.player_vel.y, self.MAX_FALL_SPEED)

        # Move player
        self.player_pos += self.player_vel
        self.player_rect.midbottom = (int(self.player_pos.x), int(self.player_pos.y))

        # Collision detection
        self.on_ground = False
        for platform in self.platforms:
            p_rect = platform['rect']
            if self.player_rect.colliderect(p_rect) and self.player_vel.y > 0:
                # Check if player was above the platform in the previous frame
                if self.player_pos.y - self.player_vel.y <= p_rect.top + 1:
                    self.player_rect.bottom = p_rect.top
                    self.player_pos.y = self.player_rect.bottom
                    self.player_vel.y = 0
                    if not self.on_ground: # Just landed
                        # sfx: PlayerLand.wav
                        self._spawn_particles(self.player_rect.midbottom, 8, (160,120,80), (-1.5,1.5), (-1,-0.5))
                    self.on_ground = True

                    if platform['type'] == 'crumble':
                        platform['state']['health'] -= 20
                        if platform['state']['health'] > 0:
                            # sfx: RockCrumble.wav
                            self._spawn_particles(p_rect.midbottom, 2, self.COLOR_STONE, (-1,1), (0,1), 5)


        # Boundary checks
        self.player_pos.x = np.clip(self.player_pos.x, self.player_rect.width / 2, self.WIDTH - self.player_rect.width / 2)
        self.player_rect.midbottom = (int(self.player_pos.x), int(self.player_pos.y))

    def _update_world_state(self):
        # Update moving platforms
        for p in self.platforms:
            if p.get('state', {}).get('is_moving'):
                target_y = p['state']['target_y']
                if abs(p['rect'].y - target_y) > 1:
                    p['rect'].y += np.sign(target_y - p['rect'].y) * 2 # Move speed
                else:
                    p['rect'].y = target_y
                    p['state']['is_moving'] = False
        
        # Update crumbling platforms
        new_platforms = []
        for p in self.platforms:
            if p['type'] == 'crumble' and p['state']['health'] <= 0:
                # sfx: PlatformBreak.wav
                self._spawn_particles(p['rect'].center, 30, self.COLOR_STONE, (-2,2), (-2,5), 10)
                continue # Remove platform
            new_platforms.append(p)
        self.platforms = new_platforms

        # Update water basins
        self.current_water_volume = 0
        for basin in self.water_basins:
            if basin['is_filling'] and basin['volume'] < basin['capacity']:
                basin['volume'] += 1.0 # Fill rate
                # sfx: WaterFillLoop.wav
                self._spawn_particles(
                    (basin['rect'].centerx, basin['rect'].top),
                    1, self.COLOR_PROGRESS_FILL, (-0.5,0.5), (1,3), 5, 10
                )
            basin['volume'] = min(basin['volume'], basin['capacity'])
            self.current_water_volume += basin['volume']

        # Update particles
        self.particles = [p for p in self.particles if p['lifetime'] > 0]
        for p in self.particles:
            p['pos'] += p['vel']
            p['vel'].y += 0.1 # Particle gravity
            p['lifetime'] -= 1
            p['radius'] *= 0.98

    def _get_observation(self):
        self._render_background()
        self._render_game_elements()
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        self.screen.fill(self.COLOR_BG_SKY)
        pygame.draw.polygon(self.screen, self.COLOR_BG_MOUNTAINS, [(0, 300), (150, 150), (300, 250), (450, 180), (640, 280), (640, 400), (0, 400)])
        pygame.draw.polygon(self.screen, (0,0,0,30), [(0,0), (640,0), (640,400), (0,400)]) # Vignette

    def _render_game_elements(self):
        # Render water basins (behind platforms)
        for basin in self.water_basins:
            fill_ratio = basin['volume'] / basin['capacity']
            water_height = int(basin['rect'].height * fill_ratio)
            if water_height > 0:
                water_rect = pygame.Rect(basin['rect'].left, basin['rect'].bottom - water_height, basin['rect'].width, water_height)
                
                # Draw a slightly darker base color
                pygame.draw.rect(self.screen, (40, 120, 225), water_rect)
                
                # Draw a wavy surface on top
                for i in range(water_rect.width // 10):
                    wave_offset = math.sin(self.steps * 0.1 + i) * 2
                    start_pos = (water_rect.left + i * 10, water_rect.top + wave_offset)
                    end_pos = (water_rect.left + (i + 1) * 10, water_rect.top + wave_offset)
                    pygame.draw.line(self.screen, (100, 180, 255), start_pos, end_pos, 3)

        # Render platforms
        for platform in self.platforms:
            color = self.COLOR_STONE if platform['type'] in ['stone', 'crumble'] else self.COLOR_EARTH
            pygame.draw.rect(self.screen, color, platform['rect'])
            pygame.draw.rect(self.screen, (0,0,0,50), platform['rect'], 2) # Outline
            if platform['type'] == 'crumble':
                crack_progress = 1 - platform['state']['health'] / 100.0
                if crack_progress > 0:
                    self._draw_cracks(platform['rect'], int(crack_progress * 10))

        # Render tool targets
        for target in self.tool_targets:
            if not target['activated']:
                pulse = (math.sin(self.steps * 0.1) + 1) / 2 * 5
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), int(target['radius'] + pulse), (*self.COLOR_TARGET_GOLD, 50))
                pygame.gfxdraw.filled_circle(self.screen, int(target['pos'].x), int(target['pos'].y), target['radius'], self.COLOR_TARGET_GOLD)

        # Render particles
        for p in self.particles:
            if p['radius'] > 1:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

        # Render player
        glow_radius = int(self.player_rect.width * 1.5)
        pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos.x), int(self.player_pos.y - self.player_rect.height/2), glow_radius, self.COLOR_PLAYER_GLOW)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, self.player_rect, border_radius=3)

    def _render_ui(self):
        # UI Background Panel
        panel = pygame.Surface((self.WIDTH, 50), pygame.SRCALPHA)
        panel.fill(self.COLOR_UI_BG)
        self.screen.blit(panel, (0, 0))

        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Water Progress Bar
        progress = self.current_water_volume / self.total_water_capacity if self.total_water_capacity > 0 else 0
        progress = np.clip(progress, 0, 1)
        bar_w, bar_h = 200, 20
        bar_x, bar_y = self.WIDTH // 2 - bar_w // 2, 15
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_EMPTY, (bar_x, bar_y, bar_w, bar_h), border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_PROGRESS_FILL, (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=5)
        progress_text = self.font_small.render("WATER RESTORED", True, self.COLOR_UI_TEXT)
        self.screen.blit(progress_text, (bar_x + bar_w / 2 - progress_text.get_width() / 2, bar_y + bar_h / 2 - progress_text.get_height() / 2))

        # Current Tool
        tool_text = self.font_large.render(f"TOOL: {self.tools[self.current_tool]}", True, self.COLOR_UI_TEXT)
        self.screen.blit(tool_text, (self.WIDTH - tool_text.get_width() - 10, 10))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "water_progress": self.current_water_volume / self.total_water_capacity if self.total_water_capacity > 0 else 0,
            "player_pos": (self.player_pos.x, self.player_pos.y),
            "current_tool": self.tools[self.current_tool]
        }
        
    def _get_dist_to_goal(self):
        # Goal is the final water basin
        if self.water_basins:
            goal_center = pygame.Vector2(self.water_basins[-1]['rect'].center)
            return self.player_pos.distance_to(goal_center)
        return 0

    def _spawn_particles(self, pos, count, color, x_vel_range, y_vel_range, radius=4, lifetime=20):
        for _ in range(count):
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(random.uniform(*x_vel_range), random.uniform(*y_vel_range)),
                'color': color,
                'radius': radius * random.uniform(0.8, 1.2),
                'lifetime': lifetime * random.uniform(0.8, 1.2)
            })
            
    def _draw_cracks(self, rect, num_cracks):
        for i in range(num_cracks):
            start = pygame.Vector2(
                random.uniform(rect.left, rect.right),
                random.uniform(rect.top, rect.bottom)
            )
            angle = random.uniform(0, 2 * math.pi)
            end = start + pygame.Vector2(math.cos(angle), math.sin(angle)) * random.randint(5, 15)
            end.x = np.clip(end.x, rect.left, rect.right)
            end.y = np.clip(end.y, rect.top, rect.bottom)
            pygame.draw.line(self.screen, (0,0,0,100), start, end, 1)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You might need to `pip install pygame`
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # Arrows: Move
    # Space: Use Tool
    # Shift: Switch Tool
    
    print(GameEnv.user_guide)
    print("Goal: Activate tools to create a path and fill the final basin with water.")

    # Game loop for human play
    running = True
    # Ensure a display is available for manual play
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.display.init()
    display_surface = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Aztec Water Temple")

    while running:
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Actions
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the screen
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        display_surface.blit(render_surface, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        env.clock.tick(env.FPS)

    env.close()