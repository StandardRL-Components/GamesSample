import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import os
import math
import os
import pygame


# Set the video driver to dummy before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
import pygame.gfxdraw


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Hold Space and press an arrow key to attack. A move or attack ends your turn."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based strategy game. Navigate the grid and defeat all the monsters before they take you down."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    TILE_SIZE = 50
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    GRID_PIXEL_WIDTH = GRID_WIDTH * TILE_SIZE
    GRID_PIXEL_HEIGHT = GRID_HEIGHT * TILE_SIZE
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_PIXEL_WIDTH) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_PIXEL_HEIGHT) // 2

    COLOR_BG = (25, 25, 35)
    COLOR_GRID = (50, 50, 60)
    COLOR_PLAYER = (50, 200, 50)
    COLOR_PLAYER_OUTLINE = (150, 255, 150)
    COLOR_MONSTER_1 = (200, 50, 50)
    COLOR_MONSTER_2 = (220, 100, 50)
    COLOR_MONSTER_3 = (240, 150, 50)
    MONSTER_COLORS = [COLOR_MONSTER_1, COLOR_MONSTER_2, COLOR_MONSTER_3]
    COLOR_ATTACK = (255, 255, 100)
    COLOR_HIT = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_OUTLINE = (10, 10, 10)
    
    PLAYER_ANIM_FRAMES = 8
    MONSTER_ANIM_FRAMES = 12
    EFFECT_FRAMES = 6
    
    MAX_STEPS = 1500

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
        # A display mode must be set for certain pygame functions to work in headless mode.
        pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        # Etc...        
        self.game_phase = "INIT"
        self.anim_timer = 0
        self.particles = []
        self.effects = []
        
        # Initialize state variables - reset() is called by the user/wrapper, not in __init__
        # self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state, for example:
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player = {
            "logic_pos": [self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1],
            "render_pos": self._grid_to_pixel([self.GRID_WIDTH // 2, self.GRID_HEIGHT - 1]),
            "health": 3,
            "max_health": 3
        }

        self.monsters = []
        occupied_positions = {tuple(self.player["logic_pos"])}
        for i in range(5):
            while True:
                pos = [self.np_random.integers(0, self.GRID_WIDTH), self.np_random.integers(0, self.GRID_HEIGHT // 2)]
                if tuple(pos) not in occupied_positions:
                    occupied_positions.add(tuple(pos))
                    break
            
            max_health = self.np_random.integers(1, 4)
            self.monsters.append({
                "id": i,
                "logic_pos": pos,
                "render_pos": self._grid_to_pixel(pos),
                "health": max_health,
                "max_health": max_health,
                "is_hit": 0
            })
        
        self.particles = []
        self.effects = []
        self.game_phase = "PLAYER_INPUT"
        self.anim_timer = 0
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action is handled in phase-specific methods
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        frame_reward = 0

        # --- Game State Machine ---
        if self.game_phase == "PLAYER_INPUT":
            frame_reward += self._handle_player_input(action)
        elif self.game_phase == "PLAYER_ANIM":
            frame_reward += self._update_player_anim()
        elif self.game_phase == "MONSTER_ANIM":
            frame_reward += self._update_monster_anim()
        
        self._update_effects_and_particles()

        # --- Termination ---
        terminated = False
        truncated = False
        if not self.game_over:
            if self.player["health"] <= 0:
                self.game_over = True
                terminated = True
                frame_reward -= 100
                self.effects.append({"type": "GAME_OVER_TEXT", "text": "DEFEAT", "timer": 60})
            elif not self.monsters:
                self.game_over = True
                terminated = True
                frame_reward += 100
                self.score += 50 # Victory bonus
                self.effects.append({"type": "GAME_OVER_TEXT", "text": "VICTORY!", "timer": 60})
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                truncated = True
        else:
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            frame_reward,
            terminated,
            truncated,
            self._get_info()
        )
    
    def _handle_player_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        if movement == 0:
            return 0

        action_type = "attack" if space_held else "move"
        
        direction = [0, 0]
        if movement == 1: direction = [0, -1]  # Up
        elif movement == 2: direction = [0, 1]   # Down
        elif movement == 3: direction = [-1, 0]  # Left
        elif movement == 4: direction = [1, 0]   # Right
        
        self.player_action = {
            "type": action_type,
            "direction": direction,
            "start_pos": self.player["logic_pos"][:],
            "end_pos": [self.player["logic_pos"][0] + direction[0], self.player["logic_pos"][1] + direction[1]]
        }
        
        if action_type == "move":
            target_pos = self.player_action["end_pos"]
            is_valid_move = (0 <= target_pos[0] < self.GRID_WIDTH and
                             0 <= target_pos[1] < self.GRID_HEIGHT and
                             all(tuple(target_pos) != tuple(m["logic_pos"]) for m in self.monsters))
            if is_valid_move:
                self.game_phase = "PLAYER_ANIM"
                self.anim_timer = self.PLAYER_ANIM_FRAMES
        elif action_type == "attack":
            self.game_phase = "PLAYER_ANIM"
            self.anim_timer = self.EFFECT_FRAMES
            # Sound: PlayerAttack.wav
            start_px = self._grid_to_pixel(self.player["logic_pos"], center=True)
            end_px = self._grid_to_pixel(self.player_action["end_pos"], center=True)
            self.effects.append({"type": "ATTACK_LINE", "start": start_px, "end": end_px, "timer": self.EFFECT_FRAMES})

        return 0

    def _update_player_anim(self):
        self.anim_timer -= 1
        
        if self.anim_timer <= 0:
            return self._finalize_player_action()
        else:
            if self.player_action["type"] == "move":
                progress = 1.0 - (self.anim_timer / self.PLAYER_ANIM_FRAMES)
                start_px = self._grid_to_pixel(self.player_action["start_pos"])
                end_px = self._grid_to_pixel(self.player_action["end_pos"])
                self.player["render_pos"] = [
                    start_px[0] + (end_px[0] - start_px[0]) * progress,
                    start_px[1] + (end_px[1] - start_px[1]) * progress
                ]
        return 0
        
    def _finalize_player_action(self):
        reward = 0
        if self.player_action["type"] == "move":
            self.player["logic_pos"] = self.player_action["end_pos"]
            self.player["render_pos"] = self._grid_to_pixel(self.player["logic_pos"])
            # Sound: PlayerMove.wav
        elif self.player_action["type"] == "attack":
            target_pos = self.player_action["end_pos"]
            for monster in self.monsters:
                if tuple(monster["logic_pos"]) == tuple(target_pos):
                    monster["health"] -= 1
                    monster["is_hit"] = self.EFFECT_FRAMES
                    reward += 0.1
                    # Sound: MonsterHit.wav
                    self._create_damage_text("-1", monster["render_pos"])
                    if monster["health"] <= 0:
                        reward += 1
                        self.score += 10
                        # Sound: MonsterDefeated.wav
                        self._create_explosion(self._grid_to_pixel(monster["logic_pos"], center=True), self.MONSTER_COLORS[monster["max_health"]-1])
                    break
        
        self.monsters = [m for m in self.monsters if m["health"] > 0]
        
        if not self.monsters or self.game_over:
             self.game_phase = "PLAYER_INPUT"
        else:
            self._start_monster_turn()
        return reward

    def _start_monster_turn(self):
        self.game_phase = "MONSTER_ANIM"
        self.anim_timer = self.MONSTER_ANIM_FRAMES
        self.monster_actions = []

        monster_positions = {tuple(m["logic_pos"]) for m in self.monsters}
        player_pos = tuple(self.player["logic_pos"])

        for monster in self.monsters:
            start_pos = monster["logic_pos"]
            dist_x = player_pos[0] - start_pos[0]
            dist_y = player_pos[1] - start_pos[1]
            
            action = {"type": "wait", "start_pos": start_pos, "end_pos": start_pos, "monster_id": monster["id"]}
            
            if abs(dist_x) + abs(dist_y) == 1:
                action["type"] = "attack"
                # Sound: PlayerHit.wav
                start_px = self._grid_to_pixel(monster["logic_pos"], center=True)
                end_px = self._grid_to_pixel(self.player["logic_pos"], center=True)
                self.effects.append({"type": "ATTACK_LINE", "start": start_px, "end": end_px, "timer": self.EFFECT_FRAMES, "color": (255,100,100)})
            elif max(abs(dist_x), abs(dist_y)) <= 3:
                moves = []
                if dist_x != 0: moves.append((np.sign(dist_x), 0))
                if dist_y != 0: moves.append((0, np.sign(dist_y)))
                if abs(dist_x) > abs(dist_y): moves.reverse()
                
                for move in moves:
                    end_pos = (start_pos[0] + move[0], start_pos[1] + move[1])
                    if end_pos not in monster_positions and end_pos != player_pos:
                        action["type"] = "move"
                        action["end_pos"] = list(end_pos)
                        monster_positions.remove(tuple(start_pos))
                        monster_positions.add(end_pos)
                        break
            
            self.monster_actions.append(action)

    def _update_monster_anim(self):
        self.anim_timer -= 1
        
        if self.anim_timer <= 0:
            return self._finalize_monster_turn()
        else:
            progress = 1.0 - (self.anim_timer / self.MONSTER_ANIM_FRAMES)
            for action in self.monster_actions:
                monster = next((m for m in self.monsters if m["id"] == action["monster_id"]), None)
                if not monster: continue
                if action["type"] == "move":
                    start_px = self._grid_to_pixel(action["start_pos"])
                    end_px = self._grid_to_pixel(action["end_pos"])
                    monster["render_pos"] = [
                        start_px[0] + (end_px[0] - start_px[0]) * progress,
                        start_px[1] + (end_px[1] - start_px[1]) * progress
                    ]
        return 0

    def _finalize_monster_turn(self):
        reward = 0
        for action in self.monster_actions:
            monster = next((m for m in self.monsters if m["id"] == action["monster_id"]), None)
            if not monster: continue

            if action["type"] == "move":
                monster["logic_pos"] = action["end_pos"]
                monster["render_pos"] = self._grid_to_pixel(monster["logic_pos"])
            elif action["type"] == "attack":
                self.player["health"] -= 1
                reward -= 0.1
                self._create_damage_text("-1", self.player["render_pos"])
                self.effects.append({"type": "SCREEN_FLASH", "color": (255,0,0,50), "timer": 5})

        self.game_phase = "PLAYER_INPUT"
        return reward

    def _update_effects_and_particles(self):
        self.particles = [p for p in self.particles if p["timer"] > 0]
        for p in self.particles:
            p["timer"] -= 1
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1
        
        self.effects = [e for e in self.effects if e["timer"] > 0]
        for e in self.effects:
            e["timer"] -= 1
            
        for m in self.monsters:
            if m["is_hit"] > 0:
                m["is_hit"] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player["health"],
            "monsters_remaining": len(self.monsters)
        }

    def _render_game(self):
        for x in range(self.GRID_WIDTH + 1):
            start = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y)
            end = (self.GRID_OFFSET_X + x * self.TILE_SIZE, self.GRID_OFFSET_Y + self.GRID_PIXEL_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.GRID_HEIGHT + 1):
            start = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            end = (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, self.GRID_OFFSET_Y + y * self.TILE_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        for monster in self.monsters:
            self._draw_character(monster, self.MONSTER_COLORS[monster["max_health"]-1])
        
        self._draw_character(self.player, self.COLOR_PLAYER, self.COLOR_PLAYER_OUTLINE)
        
        self._render_effects_and_particles()

    def _draw_character(self, char, color, outline_color=None):
        char_size = self.TILE_SIZE * 0.7
        pos_x, pos_y = char["render_pos"]
        offset = (self.TILE_SIZE - char_size) / 2
        
        rect = pygame.Rect(pos_x + offset, pos_y + offset, char_size, char_size)
        
        if outline_color:
            outline_rect = rect.inflate(4,4)
            pygame.draw.rect(self.screen, outline_color, outline_rect, border_radius=4)
            
        pygame.draw.rect(self.screen, color, rect, border_radius=4)

        if char.get("is_hit", 0) > 0:
            flash_surface = pygame.Surface((char_size, char_size), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, 180))
            self.screen.blit(flash_surface, rect.topleft)

        hp_ratio = char["health"] / char["max_health"]
        bar_width = self.TILE_SIZE * 0.8
        bar_height = 5
        bar_x = pos_x + (self.TILE_SIZE - bar_width) / 2
        bar_y = pos_y + 3
        
        pygame.draw.rect(self.screen, (80,0,0), (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0,180,0), (bar_x, bar_y, bar_width * hp_ratio, bar_height))

    def _render_effects_and_particles(self):
        for p in self.particles:
            if p["timer"] <= 0: continue
            alpha = 255 * (p["timer"] / p["lifespan"])
            if p["type"] == "spark":
                pygame.draw.circle(self.screen, p["color"], p["pos"], p["size"] * (p["timer"]/p["lifespan"]))
            elif p["type"] == "damage_text":
                self._draw_text_outlined(p["text"], self.font_small, p["color"], p["pos"], alpha=alpha)

        for e in self.effects:
            if e["timer"] <= 0: continue
            if e["type"] == "ATTACK_LINE":
                alpha = int(255 * (e["timer"] / self.EFFECT_FRAMES))
                color = e.get("color", self.COLOR_ATTACK)
                # Create a new transparent surface for the effect instead of converting the whole screen
                temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, (*color, alpha), e["start"], e["end"], width=5)
                self.screen.blit(temp_surf, (0,0))
            elif e["type"] == "SCREEN_FLASH":
                s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                alpha = 50 * (e['timer'] / 5)
                s.fill((*e['color'][:3], alpha))
                self.screen.blit(s, (0,0))
            elif e["type"] == "GAME_OVER_TEXT":
                 self._draw_text_outlined(e["text"], self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))


    def _render_ui(self):
        self._draw_text_outlined(f"SCORE: {self.score}", self.font_large, self.COLOR_TEXT, (10, 10), align="topleft")
        
        health_text = f"HP: {self.player['health']}/{self.player['max_health']}"
        self._draw_text_outlined(health_text, self.font_large, self.COLOR_TEXT, (self.SCREEN_WIDTH - 10, 10), align="topright")
        
    def _grid_to_pixel(self, grid_pos, center=False):
        px = self.GRID_OFFSET_X + grid_pos[0] * self.TILE_SIZE
        py = self.GRID_OFFSET_Y + grid_pos[1] * self.TILE_SIZE
        if center:
            px += self.TILE_SIZE // 2
            py += self.TILE_SIZE // 2
        return [px, py]

    def _draw_text_outlined(self, text, font, color, pos, align="center", alpha=255):
        base_text = font.render(text, True, color)
        outline_text = font.render(text, True, self.COLOR_TEXT_OUTLINE)
        
        if alpha < 255:
            base_text.set_alpha(alpha)
            outline_text.set_alpha(alpha)

        text_rect = base_text.get_rect()
        if align == "center":
            text_rect.center = pos
        elif align == "topleft":
            text_rect.topleft = pos
        elif align == "topright":
            text_rect.topright = pos

        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            self.screen.blit(outline_text, (text_rect.x + dx, text_rect.y + dy))
        self.screen.blit(base_text, text_rect)
        
    def _create_explosion(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "type": "spark",
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "timer": self.np_random.integers(10, 20),
                "lifespan": 20,
                "color": color,
                "size": self.np_random.integers(2, 5)
            })

    def _create_damage_text(self, text, pos):
        self.particles.append({
            "type": "damage_text",
            "text": text,
            "pos": [pos[0] + self.TILE_SIZE/2, pos[1]],
            "vel": [self.np_random.uniform(-0.5, 0.5), -1],
            "timer": 20,
            "lifespan": 20,
            "color": self.COLOR_HIT
        })